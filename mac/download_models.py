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

print("done.", flush=True)
