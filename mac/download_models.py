from huggingface_hub import snapshot_download

print("downloading stabilityai/sdxl-turbo (fp16)...", flush=True)
snapshot_download(
    "stabilityai/sdxl-turbo",
    allow_patterns=[
        "*.json",
        "*.txt",
        "tokenizer*/**",
        "scheduler/**",
        "text_encoder/*.fp16.safetensors",
        "text_encoder_2/*.fp16.safetensors",
        "unet/*.fp16.safetensors",
        "vae/*.fp16.safetensors",
        "model_index.json",
    ],
)

print("downloading madebyollin/taesdxl...", flush=True)
snapshot_download("madebyollin/taesdxl")

# --voice pulls these in on first use; fetch them up front so a live install
# does not stall on its first spoken sentence.
print("downloading facebook/roberta-hate-speech-dynabench-r4-target...", flush=True)
snapshot_download(
    "facebook/roberta-hate-speech-dynabench-r4-target",
    allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors", "*.model"],
)

print("downloading mlx-community/whisper-large-v3-turbo...", flush=True)
snapshot_download("mlx-community/whisper-large-v3-turbo")

try:
    # Only for the Silero VAD weights that gate silent chunks; the CPU
    # transcriber itself is a fallback.
    from faster_whisper import download_model

    print("downloading faster-whisper base (CPU fallback)...", flush=True)
    download_model("base")
except ImportError:
    print("faster-whisper not installed; skipping the CPU fallback", flush=True)

print("done.", flush=True)
