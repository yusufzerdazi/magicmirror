import argparse
import math
import time

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sd_mlx import StableDiffusionXL


SIZE = 512
LATENT = SIZE // 8
WINDOW = "magicmirror"


def grab_square(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = frame[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(cropped, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def to_mx(rgb_uint8):
    arr = mx.array(rgb_uint8)
    return (arr.astype(mx.float32) / 255.0) * 2.0 - 1.0


def warp(sd, image_mx, prompt, strength, steps, seed):
    latents = sd.generate_latents_from_image(
        image_mx,
        prompt,
        n_images=1,
        strength=strength,
        num_steps=steps,
        cfg_weight=0.0,
        seed=seed,
    )
    x_t = None
    for step_latent in latents:
        x_t = step_latent
        mx.eval(x_t)
    decoded = sd.decode(x_t)
    mx.eval(decoded)
    out = np.array(decoded[0])
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def fix_steps(steps, strength):
    if int(steps * strength) < 1:
        return int(math.ceil(1.0 / strength))
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="a psychedelic dream landscape, vivid colors, painterly",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--quantize", action="store_true",
                        help="8-bit quantize the UNet and text encoders")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    args.steps = fix_steps(args.steps, args.strength)

    print("loading SDXL Turbo on MLX (fp16)...", flush=True)
    sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)

    if args.quantize:
        print("quantizing text encoders + unet (8-bit)...", flush=True)
        nn.quantize(sd.text_encoder_1,
                    class_predicate=lambda _, m: isinstance(m, nn.Linear))
        nn.quantize(sd.text_encoder_2,
                    class_predicate=lambda _, m: isinstance(m, nn.Linear))
        nn.quantize(sd.unet, group_size=32, bits=8)

    sd.ensure_models_are_loaded()
    print("models loaded.", flush=True)

    prompt = args.prompt
    seed = args.seed
    print(f"prompt: {prompt!r}", flush=True)

    print("warming up...", flush=True)
    dummy = mx.zeros((SIZE, SIZE, 3), dtype=mx.float32)
    for i in range(2):
        t0 = time.time()
        warp(sd, dummy, prompt, args.strength, args.steps, seed)
        print(f"warmup {i + 1}/2 took {time.time() - t0:.2f}s", flush=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fullscreen = args.fullscreen
    cv2.setWindowProperty(
        WINDOW,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
    )

    print("controls: q=quit  f=fullscreen  p=new prompt  s=reseed", flush=True)

    frames = 0
    last_report = time.time()
    while True:
        rgb = grab_square(cap)
        if rgb is None:
            continue

        warped = warp(sd, to_mx(rgb), prompt, args.strength, args.steps, seed)
        cv2.imshow(WINDOW, cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))

        frames += 1
        now = time.time()
        if now - last_report >= 2.0:
            print(f"~{frames / (now - last_report):.1f} fps", flush=True)
            frames = 0
            last_report = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
        if key == ord("p"):
            new_prompt = input("new prompt: ").strip()
            if new_prompt:
                prompt = new_prompt
                print(f"prompt: {prompt!r}", flush=True)
        if key == ord("s"):
            seed += 1
            print(f"seed: {seed}", flush=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
