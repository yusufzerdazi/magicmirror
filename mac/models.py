"""Swappable diffusion models.

The `mischief 3.0` backend can switch between SDXL-Turbo and SDXL-Lightning at
runtime (`/reload_model/{type}`); this does the same locally, loading the new
pipeline on a background thread so the picture keeps running on the old one
until the swap is ready.

Each entry carries the step count and strength that model actually wants —
they are not interchangeable. Turbo is a 1-4 step model used here at 2;
Lightning's 8-step UNet wants 8 steps and a much lower strength.
"""

import threading
import time

import torch
from diffusers import AutoencoderTiny, AutoPipelineForImage2Image
from diffusers.utils.logging import disable_progress_bar


class ModelSpec:
    def __init__(self, name, repo, vae, steps, strength, size, note, loader=None,
                 one_step=False, control_repo=None):
        self.name = name
        self.repo = repo
        self.vae = vae
        self.steps = steps
        self.strength = strength
        self.size = size          # native resolution
        self.note = note
        # A one-step distilled model has no usable intermediate timesteps, so it
        # cannot go through the ordinary img2img schedule at all; see
        # app.warp_batch_one_step.
        self.one_step = one_step
        # A ControlNet checkpoint that takes the camera in as a line drawing.
        # This is how a one-step model can act as a mirror at all: the picture
        # is conditioning on a text-to-image pass, not a partly-denoised init.
        self.control_repo = control_repo
        self.loader = loader or (_load_controlnet if control_repo else _load_standard)

    def load(self, device, dtype):
        return self.loader(self, device, dtype)


def _finish(pipe, spec, device, dtype):
    if spec.vae:
        pipe.vae = AutoencoderTiny.from_pretrained(spec.vae, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    disable_progress_bar()
    return pipe


def _load_standard(spec, device, dtype):
    kwargs = {"torch_dtype": dtype}
    if spec.name.startswith("sdxl"):
        kwargs["variant"] = "fp16" if dtype == torch.float16 else None
    if spec.name.startswith("sdxs") or spec.name == "sd-turbo":
        kwargs["safety_checker"] = None
    pipe = AutoPipelineForImage2Image.from_pretrained(spec.repo, **kwargs)
    return _finish(pipe, spec, device, dtype)


def _load_controlnet(spec, device, dtype):
    """Base model plus its ControlNet, as the SDXS sketch demo wires it up."""
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    controlnet = ControlNetModel.from_pretrained(spec.control_repo, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        spec.repo, controlnet=controlnet, torch_dtype=dtype, safety_checker=None)
    return _finish(pipe, spec, device, dtype)


def _load_sdxl_lightning(spec, device, dtype):
    """SDXL base with ByteDance's 8-step Lightning UNet, as mischief 3.0 does."""
    from diffusers import EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    unet_path = hf_hub_download(repo_id="ByteDance/SDXL-Lightning",
                                filename="sdxl_lightning_8step_unet.safetensors")
    pipe.unet.load_state_dict(load_file(unet_path, device="cpu"))
    # Lightning's recommended scheduler.
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")
    return _finish(pipe, spec, device, dtype)


MODELS = {
    "sdxl-turbo": ModelSpec(
        "sdxl-turbo", "stabilityai/sdxl-turbo", "madebyollin/taesdxl",
        steps=2, strength=0.65, size=512,
        note="the local-run backend's model"),
    "sd-turbo": ModelSpec(
        "sd-turbo", "stabilityai/sd-turbo", "madebyollin/taesd",
        steps=2, strength=0.65, size=512,
        note="SD2.1 turbo, a third the size of SDXL and natively 512"),
    "sdxs-512": ModelSpec(
        "sdxs-512", "IDKiro/sdxs-512-dreamshaper", None,
        steps=1, strength=1.0, size=512, one_step=True,
        control_repo="IDKiro/sdxs-512-dreamshaper-sketch",
        note="one-step SD1.5 driven through its sketch ControlNet; the camera "
             "goes in as a line drawing. strength is the ControlNet scale."),
    "sdxl-lightning": ModelSpec(
        "sdxl-lightning", None, "madebyollin/taesdxl",
        steps=8, strength=0.375, size=1024,
        note="mischief 3.0's model; much slower at 8 steps",
        loader=_load_sdxl_lightning),
}

# Implemented but not offered at runtime; still loadable from code by taking a
# name back out of here. sd-turbo works well but sits between the other two
# without being distinct enough to earn a slot; sdxl-lightning needs the whole
# SDXL base checkpoint and runs at 3.2 fps against sdxs-512's 19.
DISABLED_MODELS = ("sd-turbo", "sdxl-lightning")

#: The models the config panel and --model offer, in rotation order.
AVAILABLE = tuple(name for name in MODELS if name not in DISABLED_MODELS)

DEFAULT_MODEL = "sdxl-turbo"


class ModelManager:
    """Holds the live pipeline and swaps it out on a background thread."""

    def __init__(self, device, dtype, encoder_factory, on_swap=None):
        self.device = device
        self.dtype = dtype
        self.encoder_factory = encoder_factory
        self.on_swap = on_swap
        self._lock = threading.Lock()
        self._name = None
        self._pipe = None
        self._encoder = None
        self._loading = None        # name being loaded, or None
        self._error = None

    @property
    def name(self):
        return self._name

    @property
    def loading(self):
        return self._loading

    @property
    def error(self):
        return self._error

    def current(self):
        """(pipe, encoder, spec) — safe to call from the inference thread."""
        with self._lock:
            return self._pipe, self._encoder, MODELS[self._name]

    def load_now(self, name):
        """Load synchronously; used for the first model at startup."""
        spec = MODELS[name]
        print(f"loading {name} ({spec.note})...", flush=True)
        t0 = time.time()
        pipe = spec.load(self.device, self.dtype)
        encoder = self.encoder_factory(pipe)
        with self._lock:
            self._pipe, self._encoder, self._name = pipe, encoder, name
        print(f"{name} ready in {time.time() - t0:.1f}s "
              f"(prompt encoding: {encoder.backend_name})", flush=True)
        return spec

    def switch(self, name):
        """Start loading `name`; the picture keeps running until it is ready."""
        if name not in MODELS or name == self._name or self._loading:
            return False
        self._error = None
        self._loading = name
        threading.Thread(target=self._switch_worker, args=(name,),
                         daemon=True).start()
        return True

    def _switch_worker(self, name):
        spec = MODELS[name]
        t0 = time.time()
        try:
            pipe = spec.load(self.device, self.dtype)
            encoder = self.encoder_factory(pipe)
        except Exception as e:
            print(f"could not load {name}: {e}", flush=True)
            self._error = f"{name}: {e}"
            self._loading = None
            return
        with self._lock:
            old = self._pipe
            self._pipe, self._encoder, self._name = pipe, encoder, name
        del old
        if self.device == "mps":
            torch.mps.empty_cache()
        self._loading = None
        print(f"switched to {name} in {time.time() - t0:.1f}s", flush=True)
        if self.on_swap is not None:
            self.on_swap(spec)
