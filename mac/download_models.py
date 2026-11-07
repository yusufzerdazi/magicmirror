"""Pre-fetch model weights so nothing downloads while the app is starting.

    python download_models.py                 # the default model + voice + moderation
    python download_models.py --model sdxl-lightning
    python download_models.py --all           # every switchable model

SDXL-Lightning is the one worth doing ahead of time: it pulls the whole SDXL
base checkpoint (~7GB) on top of its own UNet.
"""

import argparse

from huggingface_hub import hf_hub_download, snapshot_download

import models as registry

SDXL_PATTERNS = [
    "*.json",
    "*.txt",
    "tokenizer*/**",
    "scheduler/**",
    "text_encoder/*.fp16.safetensors",
    "text_encoder_2/*.fp16.safetensors",
    "unet/*.fp16.safetensors",
    "vae/*.fp16.safetensors",
    "model_index.json",
]


def fetch_model(name):
    spec = registry.MODELS[name]
    print(f"\n=== {name} — {spec.note}", flush=True)

    if name == "sdxl-lightning":
        print("downloading stabilityai/stable-diffusion-xl-base-1.0 (fp16, ~7GB)...",
              flush=True)
        snapshot_download("stabilityai/stable-diffusion-xl-base-1.0",
                          allow_patterns=SDXL_PATTERNS)
        print("downloading ByteDance/SDXL-Lightning 8-step UNet...", flush=True)
        hf_hub_download(repo_id="ByteDance/SDXL-Lightning",
                        filename="sdxl_lightning_8step_unet.safetensors")
    elif spec.repo:
        print(f"downloading {spec.repo}...", flush=True)
        patterns = SDXL_PATTERNS if name.startswith("sdxl") else None
        snapshot_download(spec.repo, allow_patterns=patterns)

    if spec.control_repo:
        print(f"downloading {spec.control_repo} (controlnet)...", flush=True)
        snapshot_download(spec.control_repo)
    if spec.vae:
        print(f"downloading {spec.vae} (vae)...", flush=True)
        snapshot_download(spec.vae)


def fetch_preprocessors():
    print("\n=== control image preprocessors", flush=True)
    print("downloading lllyasviel/Annotators (hed, pidinet, lineart)...", flush=True)
    snapshot_download("lllyasviel/Annotators",
                      allow_patterns=["ControlNetHED.pth", "table5_pidinet.pth",
                                      "sk_model*.pth", "netG.pth"])
    print("downloading depth-anything/Depth-Anything-V2-Small-hf...", flush=True)
    snapshot_download("depth-anything/Depth-Anything-V2-Small-hf")


def fetch_upscalers():
    import upscale

    print("\n=== upscalers", flush=True)
    for kind in upscale.WEIGHTS:
        print(f"downloading Real-ESRGAN {kind} weights...", flush=True)
        upscale.fetch_weights(kind)


def fetch_banter(model=None):
    import banter

    name = model or banter.DEFAULT_MODEL
    print(f"\n=== spoken replies\ndownloading {name}...", flush=True)
    snapshot_download(name)


def fetch_voice():
    print("\n=== voice and moderation", flush=True)
    print("downloading facebook/roberta-hate-speech-dynabench-r4-target...", flush=True)
    snapshot_download(
        "facebook/roberta-hate-speech-dynabench-r4-target",
        allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors", "*.model"],
    )
    print("downloading mlx-community/whisper-large-v3-turbo...", flush=True)
    snapshot_download("mlx-community/whisper-large-v3-turbo")
    try:
        # Also brings the Silero VAD weights that gate silent chunks.
        from faster_whisper import download_model

        print("downloading faster-whisper base (CPU fallback + VAD)...", flush=True)
        download_model("base")
    except ImportError:
        print("faster-whisper not installed; skipping the CPU fallback", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", choices=tuple(registry.MODELS),
                        help="a model to fetch; repeatable (default: the app's default)")
    parser.add_argument("--all", action="store_true", help="fetch every model")
    parser.add_argument("--no-voice", action="store_true",
                        help="skip the transcription and moderation models")
    args = parser.parse_args()

    if args.all:
        wanted = list(registry.MODELS)
    else:
        wanted = args.model or [registry.DEFAULT_MODEL]
    for name in wanted:
        fetch_model(name)
    if any(registry.MODELS[n].control_repo for n in wanted):
        fetch_preprocessors()
    fetch_upscalers()
    if not args.no_voice:
        fetch_voice()
        fetch_banter()
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
