"""Local (Apple Silicon) build of the magic mirror.

Feature parity with the hosted pair — `web/src/components/WarpPage.tsx` in front
of `backend/` — but with everything in one process: webcam in, distilled SDXL
img2img on MPS, and the same 16:9 stage, cross-fade, spinning coin, voice
transcription and scrolling transcript text on the way out.

Constants that came from the frontend or the backend are marked with where they
came from, so the two can be kept in step.
"""

import argparse
import collections
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers.utils.logging import disable_progress_bar

import banter as banter_module
import config_panel
import depth as depth_module
import devices
import dreamwave
import models as model_registry
import overlay
import upscale as upscale_module
import world as world_module
from moderation import SafetyChecker
from prompting import PromptEncoder, PromptState
from voice import BACKENDS as VOICE_BACKENDS, CHUNK_SECONDS, VoiceListener

WINDOW = "dreamwave"
ASSETS_DIR = Path(__file__).parent / "assets"

# --- frontend (WarpPage.tsx) -----------------------------------------------
SIZE = 512                       # FRAME_WIDTH / FRAME_HEIGHT
INITIAL_PROMPT = ("a mischievous cat with a third eye, matte pastel colour "
                  "pallete in a cartoon style")
MAX_BUFFER_SIZE = 16             # MAX_BUFFER_SIZE
FADE_STEP = 0.05                 # opacity decrement per 60Hz frame
SCROLL_DURATION = overlay.SCROLL_DURATION

# --- backend (settings.py / diffusion_processor.py) ------------------------
DEFAULT_BATCH_SIZE = 4           # Settings.batch_size
DEFAULT_SEED = 0                 # Settings.seed
SECOND_PROMPT = "A psychedelic landscape."   # SettingsAPI.prompt_1
TINT_INTERVAL_SECONDS = 300      # Settings.tint_interval
TINT_STRENGTH = 0.1              # Settings.tint_strength
TINT_COLOR_1 = (255, 165, 0)     # Settings.tint_color_1, orange
TINT_COLOR_2 = (255, 192, 203)   # Settings.tint_color_2, pink

NOMINAL_FRAME_SECONDS = 1.0 / 60.0
# The frontend redraws on requestAnimationFrame; pace the local loop to match.
TARGET_FRAME_SECONDS = 1.0 / 60.0


def pick_device():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def prepare_frame(frame_bgr, fit, stage_aspect=overlay.STAGE_ASPECT):
    """Webcam frame -> the RGB square the model sees.

    The model only ever sees 512x512, so the choice is which part of the camera
    to hand it and how to squeeze it in:

    * `stretch` reproduces the frontend, which squeezes the whole camera frame
      into the capture canvas (`drawImage(video, 0, 0, 512, 512)`). Combined
      with the cover-fit on the way out, this crops the top and bottom of the
      picture and leaves faces stretched vertically.
    * `crop` takes a centre square, so proportions are right, at the cost of a
      much narrower field of view once the stage crops it again.
    * `wide` takes a full 16:9 crop and squeezes that into the square, then
      un-squeezes it across the stage: the whole camera view, right
      proportions, nothing cropped away.

    The horizontal flip is the backend's `torch.flip(images, dims=[-1])`.
    """
    h, w = frame_bgr.shape[:2]
    if fit == "crop":
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        frame_bgr = frame_bgr[y0:y0 + side, x0:x0 + side]
    elif fit == "wide":
        if w / h > stage_aspect:
            crop_w = int(round(h * stage_aspect))
            x0 = (w - crop_w) // 2
            frame_bgr = frame_bgr[:, x0:x0 + crop_w]
        else:
            crop_h = int(round(w / stage_aspect))
            y0 = (h - crop_h) // 2
            frame_bgr = frame_bgr[y0:y0 + crop_h, :]
    resized = cv2.resize(frame_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    mirrored = cv2.flip(resized, 1)
    return cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)


def fit_frame_to_stage(frame_bgr, width, height, fit, squeezed=True):
    """Put a warped frame on the stage, undoing `wide`'s squeeze.

    `squeezed` says whether frames still carry that squeeze; in the 16:10 frame
    they have already been cropped to the stage's shape and must not be
    stretched again.
    """
    if fit == "wide" and squeezed:
        return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    return overlay.cover_fit(frame_bgr, width, height)


# Picture shapes. `square` shows the whole square the model renders; `16:10`
# still gives the model a centre square of the camera, then cuts a full-width
# 16:10 band out of what it draws, so the picture spans the screen with no
# borders at the sides and nothing is stretched. 16:10 is the laptop display's
# own shape, so it fills that screen edge to edge.
WIDE_FRAME = "16:10"
WIDE_ASPECT = 16.0 / 10.0
FRAMES = ("square", WIDE_FRAME)


def crop_band(rgb, aspect, bias=0.5):
    """Cut the widest `aspect` band out of `rgb`.

    `bias` places it vertically: 0 keeps the top of the square, 1 the bottom.
    """
    h, w = rgb.shape[:2]
    if w / h >= aspect:
        crop_w = int(round(h * aspect))
        x0 = (w - crop_w) // 2
        return rgb[:, x0:x0 + crop_w]
    crop_h = int(round(w / aspect))
    y0 = int(round((h - crop_h) * min(max(bias, 0.0), 1.0)))
    return rgb[y0:y0 + crop_h]


class TintCycle:
    """Alternates the two background tints on the backend's schedule."""

    def __init__(self, interval=TINT_INTERVAL_SECONDS, strength=TINT_STRENGTH,
                 color_1=TINT_COLOR_1, color_2=TINT_COLOR_2):
        self.interval = interval
        self.strength = strength
        self.color_1 = color_1
        self.color_2 = color_2
        self.use_first = True
        self.last_switch = time.time()

    def current_color(self, now):
        if now - self.last_switch >= self.interval:
            self.use_first = not self.use_first
            self.last_switch = now
        return self.color_1 if self.use_first else self.color_2

    def apply(self, rgb, now):
        if self.strength <= 0:
            return rgb
        overlay_img = np.full_like(rgb, self.current_color(now), dtype=np.uint8)
        return cv2.addWeighted(rgb, 1.0 - self.strength, overlay_img, self.strength, 0.0)


class WebcamCapture:
    def __init__(self, camera_index, fit, stage_aspect=overlay.STAGE_ASPECT,
                 buffer_size=64):
        if camera_index == "sim":
            import simulate
            self.cap = simulate.SimVideoCapture()
        else:
            self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise SystemExit(
                f"could not open camera {camera_index}.\n"
                "If the log says 'not authorized to capture video', macOS has "
                "not granted this terminal camera access: System Settings > "
                "Privacy & Security > Camera, then restart the terminal.")
        self.fit = fit
        self.stage_aspect = stage_aspect
        self.buffer = collections.deque(maxlen=buffer_size)
        self.latest_raw = None
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while not self.stop_flag.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            rgb = prepare_frame(frame, self.fit, self.stage_aspect)
            ts = time.time()
            with self.lock:
                self.buffer.append((ts, rgb))
                self.latest_raw = frame

    def raw_frame(self):
        """Most recent unprocessed camera frame, for the pre-warp preview."""
        with self.lock:
            return None if self.latest_raw is None else self.latest_raw.copy()

    def latest_spaced(self, n, span_seconds):
        """Return n frames evenly distributed across the most recent span_seconds
        of webcam time. Used so a batch's input motion matches the wall-clock
        duration the batch will occupy on screen — avoids stop-motion playback."""
        with self.lock:
            if len(self.buffer) < n:
                return None
            items = list(self.buffer)
        latest_ts = items[-1][0]
        target_start = latest_ts - span_seconds
        start_idx = 0
        for i, (ts, _) in enumerate(items):
            if ts >= target_start:
                start_idx = i
                break
        recent = items[start_idx:]
        if len(recent) < n:
            recent = items[-n:]
        if n == 1:
            return [recent[-1][1]]
        step = (len(recent) - 1) / (n - 1)
        indices = [int(round(i * step)) for i in range(n)]
        return [recent[i][1] for i in indices]

    def close(self):
        self.stop_flag.set()
        self.thread.join(timeout=1.0)
        self.cap.release()


def warp_batch(pipe, encoder, rgbs, prompt, seed, device, strength, steps):
    pils = [Image.fromarray(r) for r in rgbs]
    embeds, pooled = encoder.encode(prompt, len(pils))
    generators = [torch.Generator(device=device).manual_seed(seed) for _ in pils]
    kwargs = {}
    if pooled is not None:      # SDXL only
        kwargs["pooled_prompt_embeds"] = pooled
    out = pipe(
        prompt_embeds=embeds,
        image=pils,
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=0.0,
        generator=generators,
        **kwargs,
    ).images
    return [np.asarray(o) for o in out]


# Edge extraction for the ControlNet path. The camera is usually dim and low
# contrast, so equalising first matters more than the operator chosen: without
# it the line drawing comes out nearly blank and the model free-runs.
# Offered in the config panel, first one is the default. The rest are still
# implemented below and switch on by being added back here — `hed` and `pidinet`
# in particular read faces better than anything else tried, at about 6 fps.
EDGE_STYLES = ("depth", "depth-raw", "sobel")
# Real depth from a Kinect, via the helper. Only offered when one is attached,
# and kept separate from `depth`/`depth-raw`, which always mean depth estimated
# from the colour camera.
KINECT_STYLE = "kinect"
DISABLED_EDGE_STYLES = ("hed", "pidinet", "lineart", "lineart-anime",
                        "xdog", "canny")

# Learned boundary detectors, from the ControlNet preprocessor family. They find
# the structure of a face where a gradient operator finds only noise, which is
# what this ControlNet wants to be given.
LEARNED_EDGES = {
    "hed": "HEDdetector",
    "pidinet": "PidiNetDetector",
    "lineart": "LineartDetector",
    "lineart-anime": "LineartAnimeDetector",
}
_DETECTORS = {}


def learned_detector(style, device):
    """Lazily build one of the controlnet_aux detectors."""
    if style not in _DETECTORS:
        import controlnet_aux

        cls = getattr(controlnet_aux, LEARNED_EDGES[style])
        _DETECTORS[style] = cls.from_pretrained("lllyasviel/Annotators").to(device)
    return _DETECTORS[style]

# Estimator for the "depth" style, built on first use.
_MONO_DEPTH = []


def mono_depth(device):
    if not _MONO_DEPTH:
        _MONO_DEPTH.append(depth_module.MonoDepth(device))
    return _MONO_DEPTH[0]


def sketch_from_depth_and_frame(rgb, relative, keep=0.55, detail=1.0):
    """Subject outline from depth, detail from the colour image inside it.

    Depth alone draws every fold as a contour ring and reads as topography;
    colour edges alone lose the subject in a dark room and cover the background
    in clutter. Taking the outline from one and the detail from the other gives
    a clean figure with a face in it.
    """
    thresh = np.quantile(relative, 1.0 - keep)      # keep the nearest fraction
    mask = ((relative >= thresh) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    outline = cv2.dilate(cv2.Canny(mask, 50, 150), np.ones((3, 3), np.uint8))
    edges = sketch_from_frame(rgb, "sobel")[:, :, 0].astype(np.float32)
    edges[cv2.erode(mask, np.ones((5, 5), np.uint8)) == 0] = 0
    lines = np.clip(outline.astype(np.float32) + edges * detail, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lines, cv2.COLOR_GRAY2RGB)


def _equalised_grey(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(grey)


def sketch_from_frame(rgb, style="sobel"):
    """Camera frame -> white-on-black line drawing, the control image."""
    if style == "canny":
        grey = cv2.GaussianBlur(_equalised_grey(rgb), (5, 5), 0)
        median = np.median(grey)
        lines = cv2.Canny(grey, int(max(0, 0.67 * median)),
                          int(min(255, 1.33 * median)))
    elif style == "xdog":
        grey = _equalised_grey(rgb).astype(np.float32) / 255.0
        narrow = cv2.GaussianBlur(grey, (0, 0), 1.0)
        wide = cv2.GaussianBlur(grey, (0, 0), 1.6)
        difference = 21.0 * narrow - 20.0 * wide
        soft = np.clip(np.where(difference >= 0.02, 1.0,
                                1.0 + np.tanh(8.0 * (difference - 0.02))), 0, 1)
        lines = (255 - soft * 255).astype(np.uint8)
    else:
        grey = cv2.GaussianBlur(_equalised_grey(rgb), (5, 5), 0).astype(np.float32)
        magnitude = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3),
                             cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
        magnitude = magnitude / (magnitude.max() + 1e-6) * 255.0
        lines = np.clip(magnitude * 1.6, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lines, cv2.COLOR_GRAY2RGB)


def warp_batch_control(pipe, encoder, frames, prompt, seed, device, scale,
                       style="sobel", controls_out=None):
    """One-step generation conditioned on the camera as a line drawing.

    `style=None` means the frames are already line drawings — what the depth
    source produces — and should be passed through untouched.
    """
    embeds, _ = encoder.encode(prompt, len(frames))
    if style is None:
        sketches = list(frames)                     # already line drawings
    elif style in LEARNED_EDGES:
        try:
            detector = learned_detector(style, device)
            sketches = []
            for frame in frames:
                drawn = detector(Image.fromarray(frame), detect_resolution=SIZE,
                                 image_resolution=SIZE).convert("L")
                sketches.append(cv2.cvtColor(np.asarray(drawn), cv2.COLOR_GRAY2RGB))
        except Exception as e:
            print(f"{style} unavailable ({e}); falling back to edges", flush=True)
            _DETECTORS.pop(style, None)
            style = "sobel"
            sketches = [sketch_from_frame(f, style) for f in frames]
    elif style in ("depth", "depth-raw"):
        try:
            planes = mono_depth(device).relative(frames)
            if style == "depth-raw":
                # The depth map itself, no edge extraction. A sketch ControlNet
                # has never seen a dense grey field, but it reads it as a soft
                # shape prior and returns smoother, better-proportioned forms
                # than lines do — at the cost of the facial detail lines carry.
                sketches = [cv2.cvtColor((np.clip(d, 0, 1) * 255).astype(np.uint8),
                                         cv2.COLOR_GRAY2RGB) for d in planes]
            else:
                sketches = [sketch_from_depth_and_frame(f, d)
                            for f, d in zip(frames, planes)]
        except Exception as e:
            # Never let the estimator take the mirror down mid-show.
            print(f"depth estimation unavailable ({e}); falling back to edges",
                  flush=True)
            _MONO_DEPTH.clear()
            style = "sobel"
            sketches = [sketch_from_frame(f, style) for f in frames]
    else:
        sketches = [sketch_from_frame(f, style) for f in frames]
    if controls_out is not None:
        controls_out[:] = [(style, s) for s in sketches]
    controls = [Image.fromarray(s) for s in sketches]
    generators = [torch.Generator(device=device).manual_seed(seed) for _ in frames]
    out = pipe(
        prompt_embeds=embeds,
        image=controls,
        width=SIZE,
        height=SIZE,
        num_inference_steps=1,
        guidance_scale=0.0,
        controlnet_conditioning_scale=float(scale),
        generator=generators,
    ).images
    return [np.asarray(o) for o in out]


def warp_batch_one_step(pipe, encoder, rgbs, prompt, seed, device, blend):
    """Single UNet evaluation at the highest timestep, for one-step models.

    A model distilled to map noise to an image in one shot only has a valid
    prediction at that one timestep. diffusers' img2img has to *start* below it
    to keep any of the input, so the UNet is always asked for a prediction it
    was never trained to make and the result is under-denoised mush — visibly
    so, and the same thing happens in plain text-to-image the moment a second
    step is taken.

    Instead the image latent is blended with noise directly and denoised once at
    that timestep. `blend` is how much of the image to keep; the schedule's own
    value is about 0.07, and much above 0.3 leaves the input too far from
    anything the model saw in training and the picture falls apart.
    """
    unet, vae = pipe.unet, pipe.vae
    alphas = pipe.scheduler.alphas_cumprod.to(device)
    timestep = pipe.scheduler.config.num_train_timesteps - 1
    sqrt_alpha = alphas[timestep].sqrt()
    sqrt_one_minus = (1 - alphas[timestep]).sqrt()

    embeds, _ = encoder.encode(prompt, len(rgbs))
    # encode() expands a single embedding across the batch with stride 0, which
    # some MPS kernels reject when the UNet is called directly rather than
    # through a pipeline.
    embeds = embeds.contiguous()
    batch = np.stack(rgbs)
    x = torch.from_numpy(batch).to(device, unet.dtype).permute(0, 3, 1, 2) / 127.5 - 1.0
    with torch.no_grad():
        latents = vae.encode(x).latents
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(latents.shape, generator=generator, device=device,
                            dtype=latents.dtype)
        noisy = blend * latents + (1 - blend ** 2) ** 0.5 * noise
        predicted = unet(noisy, timestep, encoder_hidden_states=embeds).sample
        decoded = vae.decode((noisy - sqrt_one_minus * predicted) / sqrt_alpha).sample
    frames = ((decoded.permute(0, 2, 3, 1).float().clamp(-1, 1) + 1) * 127.5)
    return list(frames.cpu().numpy().astype(np.uint8))


def generate_image(pipe, encoder, spec, prompt, seed, device):
    """Plain text-to-image on whichever pipeline is loaded, for Dream World.

    The ControlNet models get a blank control image at zero weight; the img2img
    models start from noise at full strength, which amounts to the same thing.
    """
    if spec.control_repo:
        blank = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        return warp_batch_control(pipe, encoder, [blank], prompt, seed, device, 0.0,
                                  style=None)[0]
    noise = np.random.default_rng(seed).integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    if spec.one_step:
        return warp_batch_one_step(pipe, encoder, [noise], prompt, seed, device, 0.0)[0]
    return warp_batch(pipe, encoder, [noise], prompt, seed, device, 1.0,
                      max(2, spec.steps))[0]


def warmup(pipe, encoder, prompt, seed, device, strength, steps, batch_size):
    dummy = [np.zeros((SIZE, SIZE, 3), dtype=np.uint8) for _ in range(batch_size)]
    for i in range(2):
        t0 = time.time()
        warp_batch(pipe, encoder, dummy, prompt, seed, device, strength, steps)
        print(f"warmup {i + 1}/2 took {time.time() - t0:.2f}s", flush=True)


class InferenceWorker:
    def __init__(self, manager, capture, prompt_state, batch_size, settings,
                 device, output_queue, tint=None, depth_source=None,
                 depth_near=depth_module.DEFAULT_NEAR_MM,
                 depth_far=depth_module.DEFAULT_FAR_MM, upscaler=None):
        self.manager = manager
        self.upscaler = upscaler
        self.capture = capture
        self.depth_near = depth_near
        self.depth_far = depth_far
        # When present, the ControlNet's line drawing comes from depth instead
        # of the colour camera.
        self.depth_source = depth_source
        self.prompt_state = prompt_state
        self.batch_size = batch_size
        # Live settings, edited from the config panel.
        self.settings = settings
        self.device = device
        self.tint = tint
        self.output_queue = output_queue
        self.stop_flag = threading.Event()
        self.last_batch_seconds = 0.6
        # One-off GPU work (Dream World's places and objects), run between
        # batches on this thread so MPS is only ever driven from one thread.
        self.jobs = queue.Queue()
        # Person masks travel with each frame as a fourth channel when asked
        # for (Dream World composites people into a generated place).
        self.with_masks = False
        # Estimating depth just for a mask costs a frame or two a second, so
        # that only happens while the world is actually wanted; the Kinect's
        # masks are nearly free and keep learning the room regardless.
        self.mono_masks = False
        self.depth_background = world_module.DepthBackground()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, job):
        self.jobs.put(job)

    def _run_jobs(self):
        try:
            job = self.jobs.get_nowait()
        except queue.Empty:
            return
        pipe, encoder, spec = self.manager.current()
        if pipe is None:
            self.jobs.put(job)
            return
        try:
            job(pipe, encoder, spec)
        except Exception as e:
            print(f"background job failed: {e}", flush=True)

    def _masks(self, rgbs, use_depth, controls):
        """A person mask per frame, or None if there is no way to tell."""
        if use_depth:
            return self.depth_background.masks(rgbs)
        if controls and controls[0][0] == "depth-raw":
            return world_module.mask_from_relative_depth(
                [c[:, :, 0].astype(np.float32) / 255.0 for _, c in controls])
        if not self.mono_masks:
            return None
        try:
            planes = mono_depth(self.device).relative(rgbs)
            return world_module.mask_from_relative_depth(planes)
        except Exception as e:
            print(f"no person mask ({e})", flush=True)
            return None

    def _loop(self):
        while not self.stop_flag.is_set():
            self._run_jobs()
            settings = self.settings
            seed = settings.seed
            prompt = self.prompt_state.composed()
            pipe, encoder, _spec = self.manager.current()
            if pipe is None:
                time.sleep(0.05)
                continue

            # A depth sensor stands in for the camera only where depth is what
            # the model is being given. `sobel` still wants the colour image,
            # and img2img models want the colour image full stop.
            style = settings.edge_style
            use_depth = (style == KINECT_STYLE and self.depth_source is not None
                         and settings.kinect_on
                         and _spec.control_repo and self.depth_source.fresh())
            if style == KINECT_STYLE and not use_depth:
                # No live sensor: estimated depth from the camera is the nearest
                # thing, and far better than freezing.
                style = "depth-raw"
            source = self.depth_source if use_depth else self.capture
            if source is None:
                # Depth-only setup and the helper has gone quiet: wait for it.
                time.sleep(0.05)
                continue
            rgbs = source.latest_spaced(self.batch_size, self.last_batch_seconds)
            if rgbs is None:
                time.sleep(0.005)
                continue

            if self.tint is not None and settings.tint_on and not use_depth:
                now = time.time()
                rgbs = [self.tint.apply(r, now) for r in rgbs]

            t0 = time.time()
            controls = []
            try:
                if _spec.control_repo:
                    # The sensor hands over a finished depth map; everything
                    # else is built from the colour frame.
                    warped = warp_batch_control(
                        pipe, encoder, rgbs, prompt, seed, self.device,
                        settings.strength, None if use_depth else style,
                        controls_out=controls)
                elif _spec.one_step:
                    warped = warp_batch_one_step(pipe, encoder, rgbs, prompt, seed,
                                                 self.device, settings.strength)
                else:
                    warped = warp_batch(pipe, encoder, rgbs, prompt, seed,
                                        self.device, settings.strength, settings.steps)
            except Exception as e:
                print(f"inference error: {e}", flush=True)
                time.sleep(0.2)
                continue

            masks = None
            if self.with_masks:
                masks = self._masks(rgbs, use_depth, controls)

            if settings.frame == WIDE_FRAME:
                warped = [crop_band(w, WIDE_ASPECT, settings.crop_bias) for w in warped]
                if masks is not None:
                    masks = [crop_band(m, WIDE_ASPECT, settings.crop_bias) for m in masks]
            if self.upscaler is not None and settings.upscaler != "off":
                h, w = warped[0].shape[:2]
                try:
                    warped = self.upscaler(warped, settings.upscaler,
                                           upscale_module.output_size(w / h))
                except Exception as e:
                    print(f"upscale error: {e}", flush=True)
            self.last_batch_seconds = time.time() - t0

            for i, w in enumerate(warped):
                bgr = cv2.cvtColor(w, cv2.COLOR_RGB2BGR)
                if masks is not None:
                    m = masks[i]
                    if m.shape[:2] != bgr.shape[:2]:
                        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
                    bgr = np.dstack([bgr, m])
                try:
                    self.output_queue.put_nowait(bgr)
                except queue.Full:
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(bgr)
                    except (queue.Empty, queue.Full):
                        pass

    def close(self):
        self.stop_flag.set()
        self.thread.join(timeout=2.0)


class FrameStream:
    """Frame buffer and cross-fade, following the frontend's display loop.

    The page keeps a queue of decoded frames, discards half of it whenever it
    grows past MAX_BUFFER_SIZE/2, and fades the outgoing canvas out over the
    incoming one in FADE_STEP increments per animation frame.

    That fixed step is a 333ms fade, which caps the picture at ~3 fps however
    fast frames arrive, so by default the fade length is instead derived from
    the measured arrival rate and shortened further when frames are queued.
    Frames are then shown as fast as they are produced and the queue stays
    drained. Pass an explicit `fade_step` for the frontend's exact behaviour.
    """

    # Bounds on the adaptive fade: long enough to read as a fade, short enough
    # not to hold frames back.
    MIN_FADE_SECONDS = 0.02
    MAX_FADE_SECONDS = 0.4
    # How hard to correct when the queue drifts from its target depth. Low on
    # purpose — a stiff correction turns each arriving batch into a stutter.
    CATCHUP_GAIN = 0.10

    def __init__(self, source_queue, fade_step=None, batch_size=DEFAULT_BATCH_SIZE,
                 max_buffer_size=MAX_BUFFER_SIZE, clock=time.time):
        self.source_queue = source_queue
        self.clock = clock
        self.fade_step = fade_step
        # Sit about half a batch deep: enough to ride out the gap between
        # batches, little enough not to add visible latency.
        self.target_backlog = max(1.0, batch_size / 2.0)
        self.max_buffer_size = max_buffer_size
        self.pending = collections.deque()
        self.current = None
        self.incoming = None
        self.opacity = 1.0
        self.frames_shown = 0
        self.frames_dropped = 0
        # Arrival timestamps, for the adaptive fade. Frames arrive in bursts of
        # one inference batch, so the rate is measured over a window rather
        # than between consecutive frames.
        self._arrivals = collections.deque(maxlen=32)
        # Optional per-frame transform, run once as each frame arrives — used
        # to fit frames to the stage before they are blended, so a 1080p frame
        # is not cross-faded and resized again on every display tick.
        self.prepare = None

    def _drain(self):
        while True:
            try:
                frame = self.source_queue.get_nowait()
            except queue.Empty:
                break
            if self.prepare is not None:
                frame = self.prepare(frame)
            self.pending.append(frame)
            self._arrivals.append(self.clock())
        if len(self.pending) > self.max_buffer_size // 2:
            # Frontend behaviour: throw away the older half rather than lag.
            drop = len(self.pending) // 2
            for _ in range(drop):
                self.pending.popleft()
            self.frames_dropped += drop

    def arrival_interval(self):
        """Seconds between produced frames, averaged over the recent window."""
        if len(self._arrivals) < 2:
            return 0.25  # the frontend's FRAME_INTERVAL, until we know better
        span = self._arrivals[-1] - self._arrivals[0]
        if span <= 0:
            return 0.25
        return span / (len(self._arrivals) - 1)

    def _fade_duration(self):
        """How long the next cross-fade should take.

        Frames arrive a whole inference batch at a time, so the queue has to be
        used as a jitter buffer: play them out at the rate they are *produced*,
        not as fast as they can be drained. Draining hard is what makes a batch
        of four flash past and then leaves the picture frozen until the next one
        lands. So the fade is one production interval long, nudged gently by how
        far the queue is from the depth we want to sit at.
        """
        interval = self.arrival_interval()
        target = max(1.0, self.target_backlog)
        speed = 1.0 + self.CATCHUP_GAIN * (len(self.pending) - target) / target
        speed = min(max(speed, 0.6), 2.0)
        return min(max(interval / speed, self.MIN_FADE_SECONDS),
                   self.MAX_FADE_SECONDS)

    def tick(self, dt):
        """Advance the fade by `dt` seconds and return the frame to display."""
        self._drain()

        if self.incoming is None and self.pending:
            # Wait until a small buffer has built before playing out, and again
            # after a stall. Frames arrive a batch at a time, so without a
            # buffer there is nothing to smooth the gaps between batches with
            # and the picture stutters batch by batch.
            if self.fade_step is None and len(self.pending) < self.target_backlog \
                    and self.current is not None:
                return self.current
            if self.current is None:
                # Nothing to fade from: show the first frame immediately.
                self.current = self.pending.popleft()
                self.frames_shown += 1
                return self.current
            self.incoming = self.pending[0]
            self.opacity = 1.0
            if self.incoming.shape != self.current.shape:
                # The frame shape or upscaler was changed: fade from the old
                # picture resized, rather than failing to blend.
                h, w = self.incoming.shape[:2]
                if self.current.shape[:2] != (h, w):
                    self.current = cv2.resize(self.current, (w, h),
                                              interpolation=cv2.INTER_LINEAR)
                # Person masks switched on or off: match the channel count.
                cin, ccur = self.incoming.shape[2], self.current.shape[2]
                if ccur < cin:
                    self.current = np.dstack([self.current,
                                              np.full((h, w), 255, np.uint8)])
                elif ccur > cin:
                    self.current = np.ascontiguousarray(self.current[:, :, :cin])

        if self.incoming is None:
            return self.current

        if self.fade_step is not None:
            self.opacity -= self.fade_step * (dt / NOMINAL_FRAME_SECONDS)
        else:
            self.opacity -= dt / self._fade_duration()

        if self.opacity <= 0.0:
            self.current = self.incoming
            self.incoming = None
            self.opacity = 1.0
            if self.pending:
                self.pending.popleft()
            self.frames_shown += 1
            return self.current

        return cv2.addWeighted(self.current, self.opacity,
                               self.incoming, 1.0 - self.opacity, 0.0)


class LiveSettings:
    """Everything the config panel can change while the app is running."""

    def __init__(self, manager, prompt_state, args, stage_aspect):
        self.manager = manager
        self.prompt_state = prompt_state
        spec = model_registry.MODELS[manager.name]
        self.steps = args.steps if args.steps is not None else spec.steps
        self.strength = args.strength if args.strength is not None else spec.strength
        self.seed = args.seed
        self.rotation = args.rotation
        self.capture_fit = args.capture_fit
        self.has_kinect = False         # set once the depth source is up
        self.kinect_on = True           # the Kinect can be switched off live
        self.can_drop_kinect = True     # False when it is the only input
        self.edge_style = (model_registry.MODELS[manager.name].edge_style
                           or EDGE_STYLES[0])
        self.rotate_minutes = args.rotate_models
        self.tint_on = not args.no_tint
        self.background_on = args.background == "circles"
        self.fade_step = None
        # The --stage-aspect shape, used by the square frame.
        self.base_aspect = stage_aspect
        self.frame = args.frame
        self.crop_bias = args.crop_bias
        self.upscaler = args.upscaler
        self.pending_model = None
        self.on_seed = None
        self.on_framing = None
        self.on_frame = None
        self.world_mode = getattr(args, "world", "auto")
        self.depth_source = None
        self.depth_near = args.depth_near
        self.depth_far = args.depth_far

    @property
    def stage_aspect(self):
        """Shape of the picture on screen."""
        return WIDE_ASPECT if self.frame == WIDE_FRAME else self.base_aspect

    @property
    def input_aspect(self):
        """Shape of the camera/depth region the model is handed (squeezed into
        its square). The 16:10 frame crops after the model, so it takes a square."""
        return 1.0 if self.frame == WIDE_FRAME else self.base_aspect

    def cycle_frame(self, delta):
        self.frame = FRAMES[(FRAMES.index(self.frame) + delta) % len(FRAMES)]
        if self.on_frame:
            self.on_frame()

    def cycle_upscaler(self, delta):
        kinds = upscale_module.KINDS
        self.upscaler = kinds[(kinds.index(self.upscaler) + delta) % len(kinds)]

    def adjust_crop_bias(self, delta):
        self.crop_bias = round(min(max(self.crop_bias + 0.05 * delta, 0.0), 1.0), 2)

    def switch_model(self, name):
        """Start loading `name` and take its own steps and strength with it."""
        spec = model_registry.MODELS[name]
        if not self.manager.switch(name):
            return False
        self.steps = spec.steps
        self.strength = spec.strength
        preferred = self.preferred_edges(spec)
        if preferred:
            self.edge_style = preferred
        self.pending_model = None
        return True

    def next_model(self):
        """The model after the current one, in the offered order."""
        names = model_registry.AVAILABLE
        if not names:
            return None
        try:
            index = names.index(self.manager.name)
        except ValueError:
            return names[0]     # running something not in the offered list
        return names[(index + 1) % len(names)]

    def adjust_rotation(self, delta):
        choices = [0, 5, 10, 15, 30, 45, 60]
        index = (choices.index(self.rotate_minutes)
                 if self.rotate_minutes in choices else 0)
        self.rotate_minutes = choices[min(max(index + delta, 0), len(choices) - 1)]

    def set_seed(self, value):
        self.seed = int(value)
        if self.on_seed:
            self.on_seed(self.seed)

    def set_rotation(self, value):
        self.rotation = value

    def cycle_framing(self, delta):
        order = ["wide", "stretch", "crop"]
        self.capture_fit = order[(order.index(self.capture_fit) + delta) % len(order)]
        if self.on_framing:
            self.on_framing(self.capture_fit)

    def edge_styles(self):
        """What the panel offers: the Kinect first, when there is one."""
        return ((KINECT_STYLE,) + EDGE_STYLES) if self.has_kinect else EDGE_STYLES

    def preferred_edges(self, spec):
        """A depth-hungry model gets the real sensor when one is attached."""
        if spec.edge_style and spec.control_repo and self.has_kinect:
            return KINECT_STYLE
        return spec.edge_style

    def cycle_edges(self, delta):
        styles = self.edge_styles()
        current = styles.index(self.edge_style) if self.edge_style in styles else 0
        self.edge_style = styles[(current + delta) % len(styles)]

    def toggle_tint(self):
        self.tint_on = not self.tint_on

    def toggle_background(self):
        self.background_on = not self.background_on

    def cycle_world(self, delta):
        modes = ["auto", "on", "off"]
        self.world_mode = modes[(modes.index(self.world_mode) + delta) % len(modes)]
        print(f"dream world: {self.world_mode}", flush=True)

    def toggle_world(self, showing):
        """The `w` key: whatever is on screen now, switch to the other."""
        self.world_mode = "off" if showing else "on"
        print(f"dream world: {self.world_mode}", flush=True)

    def kinect_live(self):
        return self.has_kinect and self.kinect_on

    def toggle_kinect(self, _delta=1):
        if not self.has_kinect:
            print("kinect: none attached", flush=True)
            return
        if self.kinect_on and not self.can_drop_kinect:
            print("kinect: it is the only input, staying on", flush=True)
            return
        self.kinect_on = not self.kinect_on
        print(f"kinect: {'on' if self.kinect_on else 'off (camera only)'}", flush=True)

    def adjust_depth(self, near_delta, far_delta):
        """Move the Kinect's working volume live, 100mm a step."""
        near = min(max(self.depth_near + 100 * near_delta, 300), 2000)
        far = min(max(self.depth_far + 100 * far_delta, near + 400), 6000)
        self.depth_near, self.depth_far = near, far
        source = self.depth_source
        if source is not None and hasattr(source, "near"):
            source.near, source.far = near, far

    def adjust_fade(self, delta):
        steps = [None, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]
        index = steps.index(self.fade_step) if self.fade_step in steps else 0
        self.fade_step = steps[min(max(index + delta, 0), len(steps) - 1)]


class TranscriptFeed:
    """Live transcripts with the frontend's 5s lifetime and side alternation."""

    def __init__(self, lifetime=SCROLL_DURATION):
        self.lifetime = lifetime
        self.entries = collections.deque(maxlen=16)
        self.last_heard = None      # drives the idle prompt
        self.lock = threading.Lock()

    def add(self, text):
        # WarpPage picks the path from the entry's millisecond timestamp parity.
        now = time.time()
        side = "left" if int(now * 1000) % 2 == 0 else "right"
        with self.lock:
            self.entries.append((text, now, side))
            self.last_heard = now

    def live(self, now):
        with self.lock:
            return [e for e in self.entries if now - e[1] < self.lifetime]


class CommandReader:
    """Reads prompt changes from stdin without stalling the render loop.

    Stands in for the backend's `/prompt/`, `/secondprompt/` and `/blend/`
    endpoints, which have no meaning in a single-process local build.
    """

    USAGE = ("commands (type in this terminal): <text> or 'prompt <text>' | "
             "'second <text>' | 'blend <0-1>' | 'seed <n>' | 'say <text>' | "
             "'model <name>' | 'steps <n>' | 'strength <x>' | 'tune' | "
             "'world auto|on|off' | 'kinect on|off' | "
             "'frame square|16:10' | 'upscaler off|lanczos|anime|general' | 'quit'")

    def __init__(self, prompt_state, on_seed, on_quit, on_transcript=None,
                 settings=None):
        self.prompt_state = prompt_state
        self.on_seed = on_seed
        self.on_quit = on_quit
        self.on_transcript = on_transcript
        # Same switches the config panel drives, reachable without the window
        # having keyboard focus.
        self.settings = settings
        # Dreamwave's dial: a typed prompt holds it, `tune` moves it on.
        self.tuner = None
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _set_prompt(self, text):
        if self.tuner is not None:
            self.tuner.hold(text)
            print(f"prompt (dial held — 'tune' to resume): "
                  f"{self.prompt_state.composed()!r}", flush=True)
        else:
            print(f"prompt: {self.prompt_state.set_prompt(text)!r}", flush=True)

    def start(self):
        # Not gated on isatty: piped stdin is how the app gets driven from a
        # test harness or a launch script.
        if sys.stdin is not None and not sys.stdin.closed:
            self.thread.start()

    def _loop(self):
        while not self.stop_flag.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            head, _, rest = line.partition(" ")
            head_lower = head.lower()
            rest = rest.strip()
            if head_lower in ("quit", "q", "exit"):
                self.on_quit()
            elif head_lower in ("prompt", "p") and rest:
                self._set_prompt(rest)
            elif head_lower == "world" and self.settings is not None:
                if rest in ("auto", "on", "off"):
                    self.settings.world_mode = rest
                print(f"dream world: {self.settings.world_mode}", flush=True)
            elif head_lower == "kinect" and self.settings is not None:
                if rest in ("on", "off") and (rest == "on") != self.settings.kinect_on:
                    self.settings.toggle_kinect()
                print(f"kinect: {'on' if self.settings.kinect_on else 'off'}", flush=True)
            elif head_lower in ("tune", "station", "next") and self.tuner is not None:
                self.tuner.resume()
                self.tuner.next_station(reason="by hand")
            elif head_lower in ("second", "secondprompt", "o") and rest:
                print(f"prompt: {self.prompt_state.set_second_prompt(rest)!r}", flush=True)
            elif head_lower == "blend" and rest:
                try:
                    print(f"prompt: {self.prompt_state.set_blend(float(rest))!r}",
                          flush=True)
                except ValueError:
                    print("blend takes a number between 0 and 1", flush=True)
            elif head_lower == "say" and rest:
                # Same path a real transcription takes, for testing the text
                # overlay (and --voice-prompt) without a microphone.
                if self.on_transcript is not None:
                    self.on_transcript(rest)
            elif head_lower in ("model", "models") and self.settings is not None:
                names = list(model_registry.AVAILABLE)
                if not rest or rest not in names:
                    current = self.settings.manager.name
                    print(f"model: {current} — available: {', '.join(names)}",
                          flush=True)
                else:
                    spec = model_registry.MODELS[rest]
                    if self.settings.switch_model(rest):
                        print(f"loading {rest} ({spec.steps} steps, "
                              f"strength {spec.strength})...", flush=True)
                    else:
                        print("already on that model, or a load is in progress",
                              flush=True)
            elif head_lower == "steps" and rest and self.settings is not None:
                try:
                    self.settings.steps = max(1, min(int(rest), 16))
                    print(f"steps: {self.settings.steps}", flush=True)
                except ValueError:
                    print("steps takes an integer", flush=True)
            elif head_lower == "strength" and rest and self.settings is not None:
                try:
                    self.settings.strength = min(max(float(rest), 0.05), 1.0)
                    print(f"strength: {self.settings.strength}", flush=True)
                except ValueError:
                    print("strength takes a number", flush=True)
            elif head_lower == "frame" and self.settings is not None:
                if rest in FRAMES:
                    while self.settings.frame != rest:
                        self.settings.cycle_frame(+1)
                else:
                    print(f"frame: {self.settings.frame} — choose from "
                          f"{', '.join(FRAMES)}", flush=True)
            elif head_lower == "upscaler" and self.settings is not None:
                if rest in upscale_module.KINDS:
                    self.settings.upscaler = rest
                print(f"upscaler: {self.settings.upscaler} — choose from "
                      f"{', '.join(upscale_module.KINDS)}", flush=True)
            elif head_lower == "seed" and rest:
                try:
                    self.on_seed(int(rest))
                except ValueError:
                    print("seed takes an integer", flush=True)
            else:
                self._set_prompt(line)

    def close(self):
        self.stop_flag.set()


def draw_prompt_editor(stage_bgr, label, buffer):
    """Type-in-the-window prompt bar.

    In fullscreen the window holds keyboard focus, so anything typed never
    reaches the terminal — the prompt has to be editable from here.
    """
    H, W = stage_bgr.shape[:2]
    scale = max(0.5, W / 1800.0)
    bar_h = int(60 * scale)
    band = stage_bgr[0:bar_h, 0:W].astype(np.float32) * 0.25
    stage_bgr[0:bar_h, 0:W] = band.astype(np.uint8)
    text = f"{label}> {buffer}_"
    # Keep the caret on screen by showing the tail of a long prompt.
    max_chars = max(20, int(W / (18 * scale)))
    if len(text) > max_chars:
        text = "..." + text[-max_chars:]
    cv2.putText(stage_bgr, text, (int(16 * scale), int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, (0, 0, 0),
                max(3, int(4 * scale)), cv2.LINE_AA)
    cv2.putText(stage_bgr, text, (int(16 * scale), int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, (255, 255, 255),
                max(1, int(2 * scale)), cv2.LINE_AA)


def draw_preview(stage_bgr, capture, rotation, prompt_state, fps, queued,
                 depth_source=None):
    """The 'h' overlay: the frontend's hidden device/rotation/status controls.

    With depth running this shows the line drawing the model is actually being
    given, which is the thing worth looking at when framing the sensor.
    """
    H, W = stage_bgr.shape[:2]
    raw = depth_source.preview() if depth_source is not None else None
    if raw is None and capture is not None:
        raw = capture.raw_frame()
    if raw is not None:
        thumb_w = max(120, W // 8)
        thumb_h = int(round(thumb_w * raw.shape[0] / raw.shape[1]))
        thumb = cv2.resize(raw, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        x0 = W - thumb_w - 16
        stage_bgr[16:16 + thumb_h, x0:x0 + thumb_w] = thumb

    prompt_0, prompt_1, blend = prompt_state.describe()
    lines = [
        f"rotation {rotation}deg   {fps:.1f} fps   queue {queued}",
        f"prompt: {prompt_0[:70]}",
        f"second: {prompt_1[:70]}   blend {blend:.2f}",
    ]
    scale = max(0.4, W / 2200.0)
    y = 32
    for line in lines:
        cv2.putText(stage_bgr, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), max(2, int(3 * scale)), cv2.LINE_AA)
        cv2.putText(stage_bgr, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), max(1, int(1.5 * scale)), cv2.LINE_AA)
        y += int(30 * max(1.0, scale))


def parse_aspect(value):
    """`fill`, or a `W:H` ratio such as 16:9."""
    if value.lower() == "fill":
        return None
    try:
        w, h = value.replace("/", ":").split(":")
        ratio = float(w) / float(h)
        if ratio <= 0:
            raise ValueError
        return ratio
    except Exception:
        raise argparse.ArgumentTypeError("--stage-aspect must be 'fill' or W:H, e.g. 16:9")


def parse_size(value):
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("--display-size must look like 1920x1080")


_SCREEN_SIZE_CACHE = []


def _screen_size_appkit():
    """Screen width and the height a window can actually occupy.

    A HighGUI fullscreen window is placed below the menu bar, so its content
    area is the display minus the menu bar. `visibleFrame` also subtracts the
    Dock, which the window sits under, so only the menu bar is taken off here.
    """
    from AppKit import NSScreen
    screen = NSScreen.mainScreen()
    frame = screen.frame()
    visible = screen.visibleFrame()
    menu_bar = frame.size.height - visible.size.height - visible.origin.y
    return int(round(frame.size.width)), int(round(frame.size.height - menu_bar))


def _screen_size_applescript():
    """Fallback: the full display bounds, without the menu bar correction."""
    result = subprocess.run(
        ["osascript", "-e",
         'tell application "Finder" to get bounds of window of desktop'],
        capture_output=True, text=True, timeout=5,
    )
    parts = [int(v) for v in result.stdout.strip().split(",")]
    if len(parts) != 4:
        raise ValueError(result.stdout)
    return parts[2] - parts[0], parts[3] - parts[1]


def detect_screen_size(fallback=(1920, 1080)):
    """Size in points of the area a fullscreen window can paint.

    Getting this exactly right matters: HighGUI scales the image to fit the
    window while preserving its aspect ratio, so a canvas that is even slightly
    the wrong shape is letterboxed by the window in its own grey rather than
    filling the screen. Deliberately not `cv2.getWindowImageRect`, which reports
    the rect of the image currently on show and so just echoes back whatever
    was last displayed.
    """
    if _SCREEN_SIZE_CACHE:
        return _SCREEN_SIZE_CACHE[0]
    size = None
    for probe in (_screen_size_appkit, _screen_size_applescript):
        try:
            width, height = probe()
            if width > 16 and height > 16:
                size = (width, height)
                break
        except Exception:
            continue
    if size is None:
        print(f"could not detect the display size; assuming {fallback[0]}x{fallback[1]} "
              f"(override with --display-size)", flush=True)
        size = fallback
    _SCREEN_SIZE_CACHE.append(size)
    return size


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=("dreamwave", "mischief"),
                        default="dreamwave",
                        help="dreamwave (default): London Decompression 2026 — "
                             "archway, radio dial, stations, on-air voice; "
                             "mischief: the original cat-and-coin look")
    parser.add_argument("--world", choices=("auto", "on", "off"), default="auto",
                        help="dreamwave: Dream World puts people inside a generated "
                             "place with parallax and spoken objects. auto "
                             "(default) alternates mirror and world each station")
    parser.add_argument("--no-stations", action="store_true",
                        help="dreamwave: hold the prompt instead of tuning "
                             "through the stations")
    parser.add_argument("--prompt", default=None,
                        help="initial prompt (default: the frontend's INITIAL_PROMPT)")
    parser.add_argument("--second-prompt", default=SECOND_PROMPT,
                        help="prompt blended towards by --blend")
    parser.add_argument("--blend", type=float, default=0.0,
                        help="0 = first prompt, 1 = second prompt")
    parser.add_argument("--camera", default="auto", metavar="WHICH",
                        help="auto (default) prefers a USB camera, then the "
                             "built-in one, then a paired iPhone; or name a kind "
                             "(usb/builtin/iphone), an index, or 'none' to run "
                             "from depth alone")
    parser.add_argument("--depth", choices=depth_module.BACKENDS, default="off",
                        help="use a Kinect's depth as the ControlNet line "
                             "drawing instead of the colour camera. 'server' "
                             "reads from the privileged helper (sudo python "
                             "depth.py --serve), which is what to use on macOS; "
                             "'kinect'/'kinect2' open the device in-process and "
                             "need the app itself to be root")
    parser.add_argument("--edges", default=None, metavar="STYLE",
                        help="control image to start on (kinect, depth, "
                             "depth-raw, sobel); default follows the model")
    parser.add_argument("--depth-near", type=int, default=depth_module.DEFAULT_NEAR_MM,
                        metavar="MM", help="near edge of the working volume")
    parser.add_argument("--depth-far", type=int, default=depth_module.DEFAULT_FAR_MM,
                        metavar="MM", help="far edge of the working volume")
    parser.add_argument("--mic", default="auto", metavar="WHICH",
                        help="same choice for the microphone")
    parser.add_argument("--list-devices", action="store_true",
                        help="print the cameras and microphones found, then exit")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rotate-models", type=int, default=30, metavar="MINUTES",
                        help="cycle through the models on this interval; 0 is "
                             "off. The timer restarts whenever the model is "
                             "changed by hand")
    parser.add_argument("--model", choices=model_registry.AVAILABLE,
                        default=model_registry.DEFAULT_MODEL,
                        help="diffusion model; switchable at runtime in the "
                             "config panel (c)")
    parser.add_argument("--steps", type=int, default=None,
                        help="inference steps (default: whatever the model wants)")
    parser.add_argument("--strength", type=float, default=None,
                        help="img2img strength (default: whatever the model wants)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                        help="number of frames per inference batch")
    parser.add_argument("--capture-fit", choices=("wide", "stretch", "crop"),
                        default="wide",
                        help="wide (default) shows the whole camera view with the "
                             "right proportions; stretch reproduces the frontend, "
                             "which squeezes the camera into a square and ends up "
                             "cropped and vertically stretched; crop takes a centre "
                             "square, correct but tightly framed")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--windowed", action="store_true",
                        help="start in a window instead of fullscreen")
    parser.add_argument("--display-size", type=parse_size, default=None,
                        metavar="WxH", help="override the detected display size")
    parser.add_argument("--stage-aspect", type=parse_aspect, default="1:1",
                        metavar="W:H",
                        help="shape of the picture area: 1:1 (default) shows the "
                             "whole square the model renders; 16:9 matches the "
                             "hosted build; 'fill' uses the whole screen")
    parser.add_argument("--frame", choices=FRAMES, default="square",
                        help="square (default) shows the model's whole square; "
                             "16:10 crops a full-width band out of it, so the "
                             "picture spans the screen with no side borders")
    parser.add_argument("--crop-bias", type=float, default=0.5, metavar="0-1",
                        help="where the 16:10 band sits in the square: 0 top, "
                             "0.5 centre (default), 1 bottom")
    parser.add_argument("--upscaler", choices=upscale_module.KINDS, default="off",
                        help="upscale frames to 1080p before display: lanczos "
                             "(plain resize), anime (Real-ESRGAN animevideov3, "
                             "fast, suits cartoon prompts), general "
                             "(realesr-general-x4v3, sharper textures, ~2x the "
                             "cost); default off")
    parser.add_argument("--background", choices=("circles", "black"),
                        default="circles",
                        help="what surrounds the picture: the mischief 3.0 "
                             "frontend's drifting blurred circles (default), or "
                             "plain black")
    parser.add_argument("--fade-step", default="auto",
                        help="cross-fade length: 'auto' (default) derives it from "
                             "how fast frames arrive so none are held back; a "
                             f"number is the frontend's fixed opacity step per "
                             f"60Hz frame ({FADE_STEP} matches the hosted build, "
                             "1 = hard cut)")
    parser.add_argument("--no-tint", action="store_true",
                        help="disable the alternating tint blended into input frames")
    parser.add_argument("--no-coin", action="store_true",
                        help="disable the spinning mischief coin overlay")
    parser.add_argument("--no-text", action="store_true",
                        help="disable the scrolling transcript text")
    parser.add_argument("--no-compel", action="store_true",
                        help="ignore compel even if installed")
    parser.add_argument("--font", default=None,
                        help="path to the transcript font (default: bundled Sigmar)")
    parser.add_argument("--voice", action="store_true",
                        help="record the mic, transcribe it, and show the text")
    parser.add_argument("--voice-prompt", action="store_true",
                        help="also drive the prompt from what is heard (the hosted "
                             "frontend only displays transcripts)")
    parser.add_argument("--voice-playback", action="store_true",
                        help="play each recorded chunk back, as the frontend does")
    parser.add_argument("--voice-chunk", type=float, default=CHUNK_SECONDS,
                        help="seconds of audio per transcription")
    parser.add_argument("--no-banter", action="store_true",
                        help="do not answer back out loud")
    parser.add_argument("--speaker", default="auto", metavar="WHICH",
                        help="where the replies come out: auto (default) prefers "
                             "the USB handset, then the built-in speakers; or "
                             "name a kind (usb/builtin) or a `say -a` id")
    parser.add_argument("--banter-mute-mic", action="store_true",
                        help="stop listening while speaking. Only needed if the "
                             "speaker is open to the room rather than in a "
                             "handset, where the mirror would hear itself")
    parser.add_argument("--banter-voice", default=banter_module.DEFAULT_VOICE,
                        help=f"macOS voice for the replies; try "
                             f"{', '.join(banter_module.VOICES)}")
    parser.add_argument("--banter-chance", type=float, default=0.7, metavar="P",
                        help="probability of answering any given remark")
    parser.add_argument("--banter-model", default=banter_module.DEFAULT_MODEL)
    parser.add_argument("--no-banter-llm", action="store_true",
                        help="use the canned lines instead of the language model")
    parser.add_argument("--idle-after", type=float,
                        default=overlay.IDLE_AFTER_SECONDS, metavar="SECONDS",
                        help="how long to wait in silence before prompting "
                             "people to speak; 0 turns it off")
    parser.add_argument("--no-speech-gate", action="store_true",
                        help="transcribe every chunk, including ones with no "
                             "speech in them — Whisper invents text out of room "
                             "tone, so this is on by default")
    parser.add_argument("--whisper-backend", choices=VOICE_BACKENDS, default="auto",
                        help="mlx = mlx-whisper on the GPU (most accurate, and what "
                             "'auto' picks); faster = faster-whisper on the CPU, "
                             "which is what the hosted backend runs")
    parser.add_argument("--whisper-model", default=None,
                        help="model for the chosen backend: mlx takes "
                             "large-v3-turbo or base; faster takes "
                             "distil-large-v3, small, base (the hosted build's "
                             "model), or any CTranslate2 repo id")
    parser.add_argument("--dump-frames", default=None, metavar="DIR",
                        help="debug: write composited frames into DIR")
    parser.add_argument("--dump-interval", type=float, default=1.0,
                        help="debug: seconds between dumped frames")
    parser.add_argument("--headless", action="store_true",
                        help="test: run without a window (use with --dump-frames)")
    parser.add_argument("--run-for", type=float, default=0.0, metavar="SECONDS",
                        help="test: quit after this long")
    parser.add_argument("--debug-geometry", action="store_true",
                        help="debug: log the window rect and stage size")
    parser.add_argument("--no-moderation", action="store_true",
                        help="skip the hate-speech check the backend runs on "
                             "every transcription")
    parser.add_argument("--banned-words", action="store_true",
                        help="also reject transcripts containing any word from the "
                             "backend's banned list")
    return parser


def main():
    args = build_parser().parse_args()
    dream = args.theme == "dreamwave"
    if args.prompt is None:
        args.prompt = (dreamwave.STATIONS[0].prompt if dream else INITIAL_PROMPT)
    if dream:
        overlay.TEXT_COLOURS = dreamwave.TEXT_COLOURS
        banter_module.SYSTEM_PROMPT = dreamwave.BANTER_SYSTEM_PROMPT
        banter_module.CANNED = dreamwave.BANTER_CANNED
        if args.font is None:
            args.font = dreamwave.font_path()

    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if not 0.0 <= args.crop_bias <= 1.0:
        raise SystemExit("--crop-bias must be between 0 and 1")
    fade_step = None
    if str(args.fade_step).lower() != "auto":
        try:
            fade_step = float(args.fade_step)
        except ValueError:
            raise SystemExit("--fade-step must be a number or 'auto'")
        if not 0.0 < fade_step <= 1.0:
            raise SystemExit("--fade-step must be in (0, 1]")

    if args.list_devices:
        print("cameras (AVFoundation order — see README on index probing):")
        print("  ", devices.describe(devices.list_cameras()))
        print("microphones (sounddevice index):")
        print("  ", devices.describe(devices.list_microphones()))
        print("speakers (`say -a` id):")
        print("  ", devices.describe(devices.list_speakers()))
        for kind in ("auto", "usb", "builtin", "iphone"):
            index, name, _, why = devices.pick_camera(None if kind == "auto" else kind)
            print(f"  --camera {kind:8} -> "
                  + (f"cv2 index {index}: {name} ({why})" if index is not None else why))
        return

    camera_index = None
    if str(args.camera).lower() == "sim":
        camera_index = "sim"
        print("camera: simulated scene", flush=True)
    elif str(args.camera).lower() != "none":
        camera_index, camera_name, camera_kind, camera_why = devices.pick_camera(args.camera)
        if camera_index is None:
            raise SystemExit(f"no camera to use — {camera_why}. "
                             f"Found: {devices.describe(devices.list_cameras())}")
        print(f"camera: [{camera_index}] {camera_name} ({camera_kind}, {camera_why})",
              flush=True)

    screen_size = args.display_size or detect_screen_size()
    # 'fill' resolves to the display's own ratio, so the capture crop and the
    # stage agree on one number from here on.
    stage_aspect = args.stage_aspect or (screen_size[0] / screen_size[1])
    print(f"display {screen_size[0]}x{screen_size[1]}, stage aspect "
          f"{stage_aspect:.3f}", flush=True)

    device, dtype = pick_device()
    print(f"device: {device} ({dtype})", flush=True)
    disable_progress_bar()
    manager = model_registry.ModelManager(
        device, dtype,
        encoder_factory=lambda pipe: PromptEncoder(pipe, device,
                                                   use_compel=not args.no_compel))
    manager.load_now(args.model)

    prompt_state = PromptState(args.prompt, args.second_prompt, args.blend)
    settings = LiveSettings(manager, prompt_state, args, stage_aspect)
    print(f"frame: {settings.frame}, upscaler: {settings.upscaler}", flush=True)
    print(f"prompt: {prompt_state.composed()!r}", flush=True)
    print(f"batch {args.batch}, steps {settings.steps}, "
          f"strength {settings.strength}", flush=True)

    pipe, encoder, _ = manager.current()
    warmup(pipe, encoder, prompt_state.composed(), settings.seed, device,
           settings.strength, settings.steps, args.batch)

    depth_source = None
    if args.depth != "off":
        if args.depth == "server":
            depth_source = depth_module.SocketDepthSource(SIZE)
        else:
            depth_source = depth_module.DepthSource(
                args.depth, SIZE, settings.input_aspect, args.depth_near,
                args.depth_far)
        if not depth_source.try_start():
            depth_source = None
    if depth_source is not None:
        settings.has_kinect = True
        settings.edge_style = (settings.preferred_edges(
            model_registry.MODELS[manager.name]) or settings.edge_style)
        print(f"edges: {settings.edge_style}", flush=True)
    if args.edges:
        if args.edges not in settings.edge_styles():
            raise SystemExit(f"--edges {args.edges} is not available; "
                             f"choose from {', '.join(settings.edge_styles())}")
        settings.edge_style = args.edges

    capture = None
    if camera_index is not None:
        capture = WebcamCapture(camera_index, args.capture_fit,
                                settings.input_aspect)
        capture.start()
        settings.on_framing = lambda fit: setattr(capture, "fit", fit)
    settings.can_drop_kinect = capture is not None
    if camera_index is None and depth_source is None:
        raise SystemExit("--camera none needs a working --depth source")

    def frame_changed():
        # Both inputs crop to the model's input shape as frames arrive, so
        # handing them the new one is all a switch takes.
        for source in (capture, depth_source):
            if source is not None and hasattr(source, "stage_aspect"):
                source.stage_aspect = settings.input_aspect
        print(f"frame: {settings.frame}", flush=True)
    settings.on_frame = frame_changed

    upscaler = upscale_module.Upscaler(device, dtype)
    if settings.upscaler in upscale_module.WEIGHTS:
        # Load (and if need be download) now rather than on the first batch.
        upscaler.preload(settings.upscaler)

    output_queue = queue.Queue(maxsize=MAX_BUFFER_SIZE)
    worker = InferenceWorker(
        manager=manager,
        capture=capture,
        depth_source=depth_source,
        depth_near=args.depth_near,
        depth_far=args.depth_far,
        prompt_state=prompt_state,
        batch_size=args.batch,
        settings=settings,
        device=device,
        output_queue=output_queue,
        tint=TintCycle(),
        upscaler=upscaler,
    )
    worker.start()

    settings.fade_step = fade_step
    stream = FrameStream(output_queue, fade_step=fade_step,
                         batch_size=args.batch)
    # What the frames get fitted to as they arrive; kept current by the main
    # loop, since the stage changes shape with the frame setting.
    stage_fit = {"size": None}

    def fit_to_stage(frame):
        size = stage_fit["size"]
        if size is None:
            return frame
        w, h = size
        return fit_frame_to_stage(frame, w, h, settings.capture_fit,
                                  squeezed=settings.frame != WIDE_FRAME)
    stream.prepare = fit_to_stage
    transcripts = TranscriptFeed()

    tuner = None
    presence = None
    dream_world = None
    if dream:
        dream_world = world_module.DreamWorld(
            submit=worker.submit,
            generate=lambda pipe, enc, spec, prompt, seed: generate_image(
                pipe, enc, spec, prompt, seed, device),
            estimate_depth=lambda images: mono_depth(device).relative(images))
        worker.with_masks = True
        settings.depth_source = depth_source
        # Load the depth estimator now, on the inference thread, rather than
        # in the middle of the first world being built.
        worker.submit(lambda pipe, enc, spec: mono_depth(device).relative(
            [np.zeros((64, 64, 3), np.uint8)]))
        presence = dreamwave.Presence()
        if not args.no_stations:
            tuner = dreamwave.Tuner(
                prompt_state,
                on_retune=lambda seed: settings.set_seed(seed))

    banter = None

    def announce_transcript(text):
        # On screen and into the prompt first; the spoken reply is queued for
        # another thread afterwards so it cannot hold either of them up.
        transcripts.add(text)
        if args.voice_prompt:
            if tuner is not None:
                subject = dreamwave.dream_from_words(text)
                if subject:
                    tuner.go_live(subject, heard=text)
                else:
                    print("voice: heard, but not drawn onto anyone", flush=True)
            else:
                print(f"voice -> prompt: {prompt_state.set_prompt(text)!r}", flush=True)
        if banter is not None:
            banter.hear(text)

    voice = None
    if args.voice:
        checker = None
        if not args.no_moderation:
            checker = SafetyChecker(use_banned_words=args.banned_words)
            threading.Thread(target=checker.warm_up, daemon=True).start()

        mic_index, mic_name, mic_kind, mic_why = devices.pick_microphone(args.mic)
        if mic_index is None and args.mic not in (None, "auto"):
            print(f"microphone: {mic_why}; falling back to the system default",
                  flush=True)
            mic_index, mic_name, mic_kind = None, "system default", "unknown"
        print(f"microphone: {mic_name} ({mic_kind}, {mic_why})", flush=True)

        if not args.no_banter:
            speaker_id, speaker_name, speaker_kind, speaker_why = \
                devices.pick_speaker(args.speaker)
            print(f"speaker: {speaker_name} ({speaker_kind}, {speaker_why})",
                  flush=True)
            banter = banter_module.Banter(
                model=args.banter_model, voice=args.banter_voice,
                output_device=speaker_id,
                use_llm=not args.no_banter_llm, reply_chance=args.banter_chance)
            banter.start()
            print(f"banter: voice={args.banter_voice}, "
                  f"{'llm' if not args.no_banter_llm else 'canned lines'}, "
                  f"chance={args.banter_chance}", flush=True)

        voice = VoiceListener(
            announce_transcript,
            input_device=mic_index,
            mute_while=(banter.is_speaking
                        if banter is not None and args.banter_mute_mic else None),
            backend=args.whisper_backend,
            model=args.whisper_model,
            chunk_seconds=args.voice_chunk,
            gate=not args.no_speech_gate,
            safety_checker=checker,
            playback=args.voice_playback,
        )
        if not voice.try_start():
            voice = None

    idle_prompt = None
    if args.idle_after > 0:
        if dream:
            idle_prompt = overlay.IdlePrompt(
                font_path=args.font, assets_dir=ASSETS_DIR, after=args.idle_after,
                headline=dreamwave.IDLE_HEADLINE,
                suggestions=dreamwave.IDLE_SUGGESTIONS,
                accent=dreamwave.BLUSH, headline_colour=dreamwave.CREAM)
        else:
            idle_prompt = overlay.IdlePrompt(font_path=args.font, assets_dir=ASSETS_DIR,
                                             after=args.idle_after)

    layout = overlay.layout_for(settings.stage_aspect)
    layout_aspect = settings.stage_aspect
    coin = None if args.no_coin else overlay.CoinRenderer(
        ASSETS_DIR, relative_size=layout.coin_relative_size * (1.6 if dream else 1.0),
        faces=dreamwave.emblem_faces if dream else None)
    COIN_SCALE = 1.6 if dream else 1.0     # the glowing centrepiece, so bigger
    text_renderer = None if args.no_text else overlay.TranscriptRenderer(
        font_path=args.font, assets_dir=ASSETS_DIR, layout=layout)
    if text_renderer is not None and text_renderer.font_path is None:
        print("no transcript font found; text will use PIL's default face", flush=True)

    quit_requested = threading.Event()
    seed_box = [args.seed]
    settings.on_seed = lambda v: seed_box.__setitem__(0, v)

    def set_seed(value):
        seed_box[0] = value
        settings.seed = value
        print(f"seed: {value}", flush=True)

    panel = config_panel.ConfigPanel(settings)

    commands = CommandReader(prompt_state, set_seed, quit_requested.set,
                             on_transcript=announce_transcript, settings=settings)
    commands.tuner = tuner
    commands.start()

    fullscreen = not args.windowed
    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            WINDOW, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
        )
        if not fullscreen:
            size = args.display_size or (1280, 720)
            cv2.resizeWindow(WINDOW, size[0], size[1])
        cv2.waitKey(1)
    started_at = time.time()

    print("controls (with the window focused): p=type a prompt  o=second prompt  "
          "c=settings  [/]=blend  s=reseed  r=rotate  h=preview  f=fullscreen  "
          + ("t=next station  w=dream world on/off  k=kinect on/off  " if dream else "")
          + "q or esc=quit", flush=True)
    print(CommandReader.USAGE, flush=True)

    if args.background != "circles":
        background = None
    elif dream:
        background = dreamwave.DuskSky()
    else:
        background = overlay.MorphingCircles()
    arch = dreamwave.Archway() if dream else None
    static = dreamwave.Static() if dream else None
    dial = dreamwave.RadioDial() if dream else None
    glow = dreamwave.Glow() if dream else None
    ribbon = footer = None
    ribbon_for = None
    world_built_for = None
    backdrop = dreamwave.Backdrop() if dream else None
    black_sky = None
    last_presence = 0.0

    show_preview = False
    last_rotation = time.time()
    last_seen_model = manager.name
    editing = None  # (label, buffer) while typing a prompt in the window
    canvas = None
    last_stage = last_stage_key = last_stage_frame = None
    last_tick = time.time()
    frames_rendered = 0
    last_report = last_tick
    fps = 0.0
    dump_dir = None
    if args.dump_frames:
        dump_dir = Path(args.dump_frames)
        dump_dir.mkdir(parents=True, exist_ok=True)
    last_dump = 0.0
    dump_index = 0
    last_geometry = 0.0

    while not quit_requested.is_set():
        now = time.time()
        dt = min(max(now - last_tick, 1e-4), 0.25)
        last_tick = now

        # Any change of model — by hand or on the timer — restarts the clock,
        # so a manual pick gets its full interval before being moved on.
        if manager.name != last_seen_model:
            last_seen_model = manager.name
            last_rotation = now
        if settings.rotate_minutes and not manager.loading \
                and now - last_rotation >= settings.rotate_minutes * 60:
            upcoming = settings.next_model()
            if upcoming and settings.switch_model(upcoming):
                print(f"auto-switch: loading {upcoming}", flush=True)
            last_rotation = now

        screen_w, screen_h = screen_size
        if canvas is None or canvas.shape[:2] != (screen_h, screen_w):
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        if dream:
            pass        # composed below, once the stage rect is known
        elif background is not None and settings.background_on:
            background.update(now)
            np.copyto(canvas, background.render(screen_w, screen_h))
        elif not settings.background_on:
            canvas[:] = 0

        if settings.stage_aspect != layout_aspect:
            # Frame shape switched from the panel: re-derive the overlay
            # geometry for the new stage.
            layout_aspect = settings.stage_aspect
            layout = overlay.layout_for(layout_aspect)
            if coin is not None:
                coin.relative_size = layout.coin_relative_size * COIN_SCALE
            if text_renderer is not None:
                text_renderer.layout = layout
        if dream:
            sx, sy, stage_w, stage_h = dreamwave.dream_stage_rect(
                screen_w, screen_h, settings.stage_aspect)
        else:
            sx, sy, stage_w, stage_h = overlay.stage_rect(screen_w, screen_h,
                                                          settings.stage_aspect)

        if presence is not None and now - last_presence >= 0.2:
            last_presence = now
            seen = None
            if depth_source is not None and depth_source.fresh() \
                    and settings.kinect_live() \
                    and settings.edge_style == KINECT_STYLE:
                seen = depth_source.preview()
                if seen is not None:
                    presence.update_depth(seen, now)
            if seen is None and capture is not None:
                raw = capture.raw_frame()
                if raw is not None:
                    presence.update_camera(raw, now)
        if tuner is not None:
            tuner.update(now, present=presence.present,
                         arrived=presence.take_arrival())
        world_showing = False
        if dream_world is not None:
            mode = settings.world_mode
            if mode == "auto":
                want = tuner is not None and tuner.tunes % 2 == 1
            else:
                want = mode == "on"
            worker.with_masks = mode != "off"
            worker.mono_masks = want
            dream_world.active = want
            tunes = tuner.tunes if tuner is not None else 0
            if mode != "off" and tunes != world_built_for:
                # Build each station's place as it is tuned in, whether or not
                # it is on show yet, so it is ready the moment it is wanted.
                world_built_for = tunes
                place = (tuner.station().place if tuner is not None
                         else dreamwave.STATIONS[0].place)
                dream_world.request_backdrop(place)
            dream_world.update(now, dt, presence)
            world_showing = want and dream_world.ready()
        stage_fit["size"] = (stage_w, stage_h)
        if stream.fade_step != settings.fade_step:
            stream.fade_step = settings.fade_step
        frame = stream.tick(dt)
        if frame is None:
            # Frontend behaviour: the raw camera shows through until the first
            # warped frame lands. With no colour camera, show what the model is
            # being given instead.
            raw = capture.raw_frame() if capture is not None else None
            if raw is None and depth_source is not None:
                raw = depth_source.preview()
            frame = raw if raw is not None else np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

        # Between cross-fades the same frame comes back every tick; at 1080p
        # re-fitting it 60 times a second is wasted work.
        stage_key = (id(frame), stage_w, stage_h, settings.capture_fit,
                     settings.frame, settings.rotation)
        if stage_key != last_stage_key:
            if frame.shape[:2] == (stage_h, stage_w):
                stage = frame       # already fitted on arrival
            else:
                stage = fit_frame_to_stage(frame, stage_w, stage_h,
                                           settings.capture_fit,
                                           squeezed=settings.frame != WIDE_FRAME)
            stage = overlay.rotate_stage(stage, settings.rotation)
            last_stage, last_stage_key = stage, stage_key
            last_stage_frame = frame     # keeps id(frame) from being reused
        stage = last_stage

        if dream:
            person, person_alpha = stage, None
            if stage.ndim == 3 and stage.shape[2] == 4:
                person, person_alpha = stage[:, :, :3], stage[:, :, 3]
            if world_showing:
                pic = dream_world.render(np.ascontiguousarray(person), person_alpha, now)
            else:
                pic = person
            level = tuner.static_level(now) if tuner is not None else 0.0
            shown = static.apply(np.ascontiguousarray(pic), level, now) if level > 0 else pic
            arch.prepare(screen_w, screen_h, (sx, sy, stage_w, stage_h))
            if ribbon_for != (screen_w, screen_h, stage_w):
                ribbon_for = (screen_w, screen_h, stage_w)
                ribbon = np.asarray(dreamwave.ribbon_image(min(stage_w * 0.62,
                                                               screen_w * 0.9)))
                footer = dreamwave.footer_image(screen_h)
            if background is not None and settings.background_on:
                if hasattr(background, "set_portal"):
                    background.set_portal((sx, sy, stage_w, stage_h))
                background.update(now)
                sky = background.render(screen_w, screen_h)
            else:
                if black_sky is None or black_sky.shape[:2] != (screen_h, screen_w):
                    black_sky = np.zeros((screen_h, screen_w, 3), np.uint8)
                sky = black_sky
            fh, fw = footer.shape[:2]
            extras = [(footer, screen_w - fw - 12, screen_h - fh - 10)]
            if tuner is not None or presence is not None:
                dw, dh = dial.size_for(screen_w, screen_h, sx)
                if tuner is not None:
                    freq, name, on_air = tuner.display(now)
                else:
                    freq, name, on_air = 88.1, "private channel", False
                dial_img = dial.render(dw, dh, freq, name, on_air,
                                       presence.level if presence is not None else 0.0,
                                       now)
                pad = max(10, int(screen_h * 0.025))
                extras.append((dial_img, pad, screen_h - dh - pad))
            np.copyto(canvas, backdrop.compose(sky, arch, extras))
            arch.paste(canvas, arch.shade(np.ascontiguousarray(shown)),
                       (sx, sy, stage_w, stage_h))
        else:
            canvas[sy:sy + stage_h, sx:sx + stage_w] = stage[:, :, :3]

        # Text and coin go on after the stage, across the whole screen, so
        # transcripts sweep in over the background instead of being clipped to
        # the picture. The coin stays anchored to the stage, above the text as
        # it is in the frontend.
        if text_renderer is not None:
            live = transcripts.live(now)
            if live:
                text_renderer.draw(canvas, live, now,
                                   stage_rect=(sx, sy, stage_w, stage_h),
                                   coin_relative_size=(coin.relative_size
                                                       if coin is not None else 0))
        if idle_prompt is not None and voice is not None:
            idle_prompt.draw(canvas, now, transcripts.last_heard,
                             rect=(sx, sy, stage_w, stage_h))
        if coin is not None:
            if glow is not None:
                gx, gy, gsize = overlay.coin_centre((sx, sy, stage_w, stage_h),
                                                    coin.relative_size)
                on_air = tuner is not None and tuner.display(now)[2]
                glow.draw(canvas, gx, gy, gsize * 0.9, now,
                          boost=1.0 if on_air else 0.0)
            coin.draw(canvas, now, rect=(sx, sy, stage_w, stage_h))
        if dream:
            dreamwave.blit_rgba(canvas, ribbon, sx + stage_w // 2 - ribbon.shape[1] // 2,
                                max(0, sy - int(ribbon.shape[0] * 0.42)))
        if show_preview:
            draw_preview(canvas, capture, settings.rotation, prompt_state, fps,
                         len(stream.pending), depth_source)
        if editing is not None:
            draw_prompt_editor(canvas, editing[0], editing[1])
        panel.draw(canvas)

        if not args.headless:
            cv2.imshow(WINDOW, canvas)
        if args.run_for and now - started_at >= args.run_for:
            quit_requested.set()

        if dump_dir is not None and now - last_dump >= args.dump_interval:
            last_dump = now
            cv2.imwrite(str(dump_dir / f"frame_{dump_index:04d}.png"), canvas)
            dump_index += 1
        if args.debug_geometry and now - last_geometry >= 1.0:
            last_geometry = now
            try:
                rect = cv2.getWindowImageRect(WINDOW)
            except Exception as e:
                rect = f"unavailable ({e})"
            print(f"geometry: canvas={screen_w}x{screen_h} "
                  f"stage={stage_w}x{stage_h}@{sx},{sy} windowImageRect={rect}",
                  flush=True)

        # waitKey doubles as the frame pacer: sleep out the rest of the tick
        # while still pumping window events.
        remaining = TARGET_FRAME_SECONDS - (time.time() - now)
        # waitKeyEx, not waitKey: the arrow keys the panel uses do not survive
        # being masked to a byte.
        if args.headless:
            time.sleep(max(0.001, remaining))
            raw_key = -1
        else:
            raw_key = cv2.waitKeyEx(max(1, int(round(remaining * 1000))))
        key = raw_key & 0xFF if 0 <= raw_key < 0x110000 else -1
        frames_rendered += 1

        if raw_key != -1 and panel.handle(raw_key):
            continue

        if key != 255 and editing is not None:
            label, buffer = editing
            if key in (13, 10):                      # return: commit
                if buffer.strip():
                    if label == "prompt" and tuner is not None:
                        tuner.hold(buffer.strip())
                        composed = prompt_state.composed()
                    elif label == "prompt":
                        composed = prompt_state.set_prompt(buffer.strip())
                    else:
                        composed = prompt_state.set_second_prompt(buffer.strip())
                    print(f"prompt: {composed!r}", flush=True)
                editing = None
            elif key == 27:                          # escape: cancel
                editing = None
            elif key in (8, 127):                    # backspace
                editing = (label, buffer[:-1])
            elif 32 <= key < 127:
                editing = (label, buffer + chr(key))
        elif key == ord("q") or key == 27:      # escape, once nothing owns it
            quit_requested.set()
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
            canvas = None
        elif key == ord("c"):
            panel.visible = True
        elif key == ord("h"):
            show_preview = not show_preview
        elif key == ord("p"):
            editing = ("prompt", "")
        elif key == ord("o"):
            editing = ("second", "")
        elif key == ord("r"):
            settings.set_rotation((settings.rotation + 90) % 360)
            print(f"rotation: {settings.rotation}", flush=True)
        elif key == ord("s"):
            set_seed(seed_box[0] + 1)
        elif key == ord("w") and dream_world is not None:
            settings.toggle_world(world_showing)
        elif key == ord("k"):
            settings.toggle_kinect()
        elif key == ord("t") and tuner is not None:
            tuner.resume()
            tuner.next_station(reason="by hand")
        elif key == ord("["):
            print(f"prompt: {prompt_state.set_blend(prompt_state.describe()[2] - 0.1)!r}",
                  flush=True)
        elif key == ord("]"):
            print(f"prompt: {prompt_state.set_blend(prompt_state.describe()[2] + 0.1)!r}",
                  flush=True)

        if now - last_report >= 2.0:
            elapsed = now - last_report
            fps = frames_rendered / elapsed
            print(f"~{fps:.0f} fps display, {stream.frames_shown / elapsed:.1f} fps warped, "
                  f"queue {len(stream.pending)}, dropped {stream.frames_dropped}",
                  flush=True)
            frames_rendered = 0
            stream.frames_shown = 0
            stream.frames_dropped = 0
            last_report = now

    commands.close()
    if banter is not None:
        banter.close()
    if voice is not None:
        voice.close()
    worker.close()
    if depth_source is not None:
        depth_source.close()
    if capture is not None:
        capture.close()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
