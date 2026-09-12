"""Diagnostic: feed 4 IDENTICAL webcam frames through the batched img2img pipe
with the same setup app.py uses. Saves the 4 outputs and the input to /tmp,
prints pairwise mean-abs-diff between outputs.

If outputs are ~identical → batching itself is uniform; the in-app effect
must be input-variance driven.
If outputs differ in a clear progression → batched img2img has a per-position
effect even with identical inputs, and we have a real bug to chase."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from diffusers.utils.logging import disable_progress_bar


SIZE = 512
OUT_DIR = Path("/tmp/mm_diag")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="elephants all the way down, 4k, photorealistic, detailed, intricate",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"device={device} dtype={dtype}")

    print("loading pipeline...")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl",
        torch_dtype=dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    disable_progress_bar()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    print("warming camera up (10 frames)...")
    for _ in range(10):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("camera read failed")

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = frame[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(cropped, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(OUT_DIR / "input.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"saved {OUT_DIR / 'input.png'}")

    pil = Image.fromarray(rgb)

    def make_gens():
        return [torch.Generator(device=device).manual_seed(args.seed)
                for _ in range(args.batch)]

    print("warmup batch x2...")
    for _ in range(2):
        pipe(prompt=[args.prompt] * args.batch,
             image=[pil] * args.batch,
             num_inference_steps=args.steps,
             strength=args.strength,
             guidance_scale=0.0,
             generator=make_gens())

    print(f"running real batch={args.batch} with identical inputs...")
    t0 = time.time()
    out = pipe(
        prompt=[args.prompt] * args.batch,
        image=[pil] * args.batch,
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=0.0,
        generator=make_gens(),
    ).images
    print(f"  took {time.time() - t0:.2f}s")

    arrs = []
    for i, img in enumerate(out):
        arr = np.asarray(img)
        arrs.append(arr.astype(np.float32))
        path = OUT_DIR / f"output_{i}.png"
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        print(f"saved {path}")

    print("\npairwise mean-abs pixel diff (0–255 scale):")
    for i in range(args.batch):
        for j in range(i + 1, args.batch):
            d = float(np.abs(arrs[i] - arrs[j]).mean())
            print(f"  output_{i} vs output_{j}: {d:.3f}")

    print("\nopen /tmp/mm_diag/ to compare visually.")


if __name__ == "__main__":
    main()
