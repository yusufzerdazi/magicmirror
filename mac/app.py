import argparse
import collections
import math
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from diffusers.utils.logging import disable_progress_bar


SIZE = 512
WINDOW = "magicmirror"

# Tint: alternating background tint blended into webcam frames pre-diffusion
TINT_INTERVAL_SECONDS = 300
TINT_STRENGTH = 0.1
TINT_COLOR_A = (255, 165, 0)    # orange (RGB)
TINT_COLOR_B = (255, 192, 203)  # pink (RGB)

# Mischief coin: 3D-style Y-axis spin with two color-tinted faces
COIN_REL_SIZE = 0.08
COIN_ROTATION_SECONDS = 3.0
COIN_BOTTOM_MARGIN_REL = 0.04
COIN_BLUE_RGB = (34, 161, 236)   # #22A1EC, matches the web frontend
COIN_PINK_RGB = (255, 186, 241)  # #FFBAF1

# Voice transcripts displayed as fading text near top of frame
VOICE_SAMPLE_RATE = 16000
VOICE_CHUNK_SECONDS = 5
TRANSCRIPT_LIFETIME_SECONDS = 5
WHISPER_MODEL = "mlx-community/whisper-tiny.en"

ASSETS_DIR = Path(__file__).parent / "assets"
MISCHIEF_PNG = ASSETS_DIR / "mischief.png"
MISCHIEF_JPG = ASSETS_DIR / "mischief.jpg"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_pipeline(device, dtype):
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
    return pipe


def to_square_rgb(frame):
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = frame[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(cropped, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def current_tint_color(now):
    use_a = (int(now / TINT_INTERVAL_SECONDS) % 2) == 0
    return TINT_COLOR_A if use_a else TINT_COLOR_B


def apply_tint_rgb(rgb, color, strength):
    if strength <= 0:
        return rgb
    overlay = np.full_like(rgb, color, dtype=np.uint8)
    return cv2.addWeighted(rgb, 1.0 - strength, overlay, strength, 0.0)


def load_mischief_alpha():
    """Return a single-channel alpha mask of the mischief logo (uint8, shape filled with 255)."""
    if MISCHIEF_PNG.exists():
        img = cv2.imread(str(MISCHIEF_PNG), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3 and img.shape[2] == 4:
                return img[:, :, 3]
            if img.ndim == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                return alpha
    if MISCHIEF_JPG.exists():
        img = cv2.imread(str(MISCHIEF_JPG))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
            return alpha
    canvas_size = 256
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    text = "M"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 6.0
    thickness = 12
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = (canvas_size - text_size[0]) // 2
    y = (canvas_size + text_size[1]) // 2
    cv2.putText(canvas, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
    return canvas


class CoinRenderer:
    """3D-style spinning coin with two color-tinted faces."""

    def __init__(self, alpha_template, blue_rgb, pink_rgb):
        self.alpha_template = alpha_template
        self.blue_rgb = blue_rgb
        self.pink_rgb = pink_rgb
        self._cached_size = 0
        self._face_blue = None
        self._face_pink = None

    @staticmethod
    def _make_face(alpha, rgb):
        h, w = alpha.shape[:2]
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[:, :, 0] = rgb[2]  # B
        bgra[:, :, 1] = rgb[1]  # G
        bgra[:, :, 2] = rgb[0]  # R
        bgra[:, :, 3] = alpha
        return bgra

    def _ensure_size(self, size):
        if size == self._cached_size:
            return
        alpha = cv2.resize(self.alpha_template, (size, size),
                           interpolation=cv2.INTER_AREA)
        self._face_blue = self._make_face(alpha, self.blue_rgb)
        self._face_pink = self._make_face(alpha, self.pink_rgb)
        self._cached_size = size

    def render(self, size, t):
        """Return a BGRA image of the coin at this moment in time, or None when edge-on."""
        self._ensure_size(size)
        angle_rad = (t / COIN_ROTATION_SECONDS) * 2 * math.pi
        cos_a = math.cos(angle_rad)
        if abs(cos_a) < 0.01:
            return None
        face = self._face_blue if cos_a >= 0 else self._face_pink
        new_w = max(1, int(size * abs(cos_a)))
        return cv2.resize(face, (new_w, size), interpolation=cv2.INTER_AREA)


def composite_bgra(base_bgr, overlay_bgra, x, y):
    H, W = base_bgr.shape[:2]
    h, w = overlay_bgra.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)
    overlay = overlay_bgra[oy0:oy1, ox0:ox1]
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    base_region = base_bgr[y0:y1, x0:x1].astype(np.float32)
    blended = base_region * (1.0 - alpha) + overlay[:, :, :3].astype(np.float32) * alpha
    base_bgr[y0:y1, x0:x1] = blended.astype(np.uint8)


def draw_transcripts(bgr, transcripts, now):
    H, W = bgr.shape[:2]
    line_y = int(H * 0.10)
    font = cv2.FONT_HERSHEY_DUPLEX
    base_scale = max(0.6, W / 800.0)
    thickness = max(1, int(2 * base_scale))
    for text, ts in list(transcripts):
        age = now - ts
        if age > TRANSCRIPT_LIFETIME_SECONDS:
            continue
        fade = max(0.0, 1.0 - age / TRANSCRIPT_LIFETIME_SECONDS)
        text_size, _ = cv2.getTextSize(text, font, base_scale, thickness)
        x = max(8, (W - text_size[0]) // 2)
        cv2.putText(bgr, text, (x, line_y), font, base_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        white = int(255 * fade)
        cv2.putText(bgr, text, (x, line_y), font, base_scale,
                    (white, white, white), thickness, cv2.LINE_AA)
        line_y += int(text_size[1] * 1.6)


class WebcamCapture:
    def __init__(self, camera_index, buffer_size=64):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open camera {camera_index}")
        self.buffer = collections.deque(maxlen=buffer_size)
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
            rgb = to_square_rgb(frame)
            ts = time.time()
            with self.lock:
                self.buffer.append((ts, rgb))

    def latest_n(self, n):
        with self.lock:
            if len(self.buffer) < n:
                return None
            return [item[1] for item in list(self.buffer)[-n:]]

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


_PROMPT_EMBEDS_CACHE = {}


def encode_prompt_tiled(pipe, prompt, batch_size, device):
    """Encode the prompt once at batch_size=1 and tile to the requested batch.

    Workaround for an MPS-specific bug where pipe.encode_prompt with batched
    identical prompts (`prompt=[p]*N`) produces non-uniform embeddings — batch
    position 0 differs from positions 1..N-1, which in turn match each other.
    The downstream effect is that output[0] looks meaningfully different from
    output[1..N-1] even when all inputs and noise are identical. Encoding once
    and repeating sidesteps the bug entirely. See diag_batch.py for the test."""
    key = (prompt, batch_size)
    cached = _PROMPT_EMBEDS_CACHE.get(key)
    if cached is not None:
        return cached
    with torch.no_grad():
        embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    embeds = embeds.repeat(batch_size, 1, 1)
    pooled = pooled.repeat(batch_size, 1)
    _PROMPT_EMBEDS_CACHE[key] = (embeds, pooled)
    return embeds, pooled


def warp_batch(pipe, rgbs, prompt, seed, device, strength, steps):
    pils = [Image.fromarray(r) for r in rgbs]
    embeds, pooled = encode_prompt_tiled(pipe, prompt, len(pils), device)
    generators = [torch.Generator(device=device).manual_seed(seed) for _ in pils]
    out = pipe(
        prompt_embeds=embeds,
        pooled_prompt_embeds=pooled,
        image=pils,
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=0.0,
        generator=generators,
    ).images
    return [np.asarray(o) for o in out]


def warmup(pipe, prompt, seed, device, strength, steps, batch_size):
    dummy = [np.zeros((SIZE, SIZE, 3), dtype=np.uint8) for _ in range(batch_size)]
    for i in range(2):
        t0 = time.time()
        warp_batch(pipe, dummy, prompt, seed, device, strength, steps)
        print(f"warmup {i + 1}/2 took {time.time() - t0:.2f}s", flush=True)


class InferenceWorker:
    def __init__(self, pipe, capture, batch_size, steps, strength, device,
                 initial_prompt, initial_seed, output_queue, apply_tint=True):
        self.pipe = pipe
        self.capture = capture
        self.batch_size = batch_size
        self.steps = steps
        self.strength = strength
        self.device = device
        self.apply_tint = apply_tint
        self._prompt = initial_prompt
        self._seed = initial_seed
        self.lock = threading.Lock()
        self.output_queue = output_queue
        self.stop_flag = threading.Event()
        self.last_batch_seconds = 0.6
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def update_prompt(self, prompt):
        with self.lock:
            self._prompt = prompt

    def update_seed(self, seed):
        with self.lock:
            self._seed = seed

    def _loop(self):
        while not self.stop_flag.is_set():
            with self.lock:
                seed = self._seed
                prompt = self._prompt

            rgbs = self.capture.latest_spaced(self.batch_size, self.last_batch_seconds)
            if rgbs is None:
                time.sleep(0.005)
                continue

            if self.apply_tint:
                color = current_tint_color(time.time())
                rgbs = [apply_tint_rgb(r, color, TINT_STRENGTH) for r in rgbs]

            t0 = time.time()
            warped = warp_batch(self.pipe, rgbs, prompt, seed, self.device,
                                self.strength, self.steps)
            batch_seconds = time.time() - t0
            self.last_batch_seconds = batch_seconds

            for w in warped:
                bgr = cv2.cvtColor(w, cv2.COLOR_RGB2BGR)
                try:
                    self.output_queue.put_nowait((bgr, batch_seconds))
                except queue.Full:
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.output_queue.put_nowait((bgr, batch_seconds))
                    except queue.Full:
                        pass

    def close(self):
        self.stop_flag.set()
        self.thread.join(timeout=2.0)


class VoiceListener:
    def __init__(self, on_transcript):
        self.on_transcript = on_transcript
        self.stop_flag = threading.Event()
        self.thread = None

    def try_start(self):
        try:
            import sounddevice  # noqa: F401
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            print(f"voice disabled — install sounddevice and mlx-whisper to enable ({e})",
                  flush=True)
            return False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def _loop(self):
        import sounddevice as sd
        import mlx_whisper

        print(f"voice: ready (model={WHISPER_MODEL}, chunk={VOICE_CHUNK_SECONDS}s)",
              flush=True)
        while not self.stop_flag.is_set():
            try:
                samples = int(VOICE_SAMPLE_RATE * VOICE_CHUNK_SECONDS)
                audio = sd.rec(samples, samplerate=VOICE_SAMPLE_RATE,
                               channels=1, dtype="float32")
                sd.wait()
                if self.stop_flag.is_set():
                    return
                result = mlx_whisper.transcribe(audio.flatten(),
                                                path_or_hf_repo=WHISPER_MODEL)
                text = (result.get("text") or "").strip()
                if text:
                    self.on_transcript(text)
            except Exception as e:
                print(f"voice error: {e}", flush=True)
                time.sleep(0.5)

    def close(self):
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="pigs all the way down, 4k, photorealistic, detailed, intricate",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=4,
                        help="number of frames per inference batch")
    parser.add_argument("--windowed", action="store_true",
                        help="start in a window instead of fullscreen")
    parser.add_argument("--no-tint", action="store_true",
                        help="disable alternating tint blend on input frames")
    parser.add_argument("--no-coin", action="store_true",
                        help="disable the spinning mischief coin overlay")
    parser.add_argument("--voice", action="store_true",
                        help="record mic audio, transcribe, and use as prompt")
    parser.add_argument("--temporal-blend", type=float, default=0.0,
                        help="EMA-blend each output with the previous frame "
                             "(0=off, ~0.3 reduces shimmer, 1=freeze)")
    args = parser.parse_args()

    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if not 0.0 <= args.temporal_blend <= 1.0:
        raise SystemExit("--temporal-blend must be in [0, 1]")

    device, dtype = pick_device()
    print(f"loading pipeline on {device} ({dtype})...", flush=True)
    pipe = load_pipeline(device, dtype)
    disable_progress_bar()
    print("pipeline loaded.", flush=True)

    prompt = args.prompt
    print(f"prompt: {prompt!r}", flush=True)
    print(f"batch size: {args.batch}", flush=True)

    warmup(pipe, prompt, args.seed, device, args.strength, args.steps, args.batch)

    capture = WebcamCapture(args.camera)
    capture.start()

    output_queue = queue.Queue(maxsize=args.batch * 2)
    worker = InferenceWorker(
        pipe=pipe,
        capture=capture,
        batch_size=args.batch,
        steps=args.steps,
        strength=args.strength,
        device=device,
        initial_prompt=prompt,
        initial_seed=args.seed,
        output_queue=output_queue,
        apply_tint=not args.no_tint,
    )
    worker.start()

    transcripts = collections.deque(maxlen=8)
    transcripts_lock = threading.Lock()

    voice = None
    if args.voice:
        def on_transcript(text):
            with transcripts_lock:
                transcripts.append((text, time.time()))
            worker.update_prompt(text)
            print(f"voice → prompt: {text!r}", flush=True)
        voice = VoiceListener(on_transcript)
        if not voice.try_start():
            voice = None

    coin = None
    if not args.no_coin:
        coin = CoinRenderer(load_mischief_alpha(), COIN_BLUE_RGB, COIN_PINK_RGB)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fullscreen = not args.windowed
    cv2.setWindowProperty(
        WINDOW,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
    )

    print("controls: q=quit  f=fullscreen  p=new prompt  s=reseed", flush=True)

    seed = args.seed
    frames_displayed = 0
    last_report = time.time()
    quit_requested = False
    last_diffused_bgr = None

    while not quit_requested:
        try:
            bgr, batch_seconds = output_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if args.temporal_blend > 0.0 and last_diffused_bgr is not None \
                and last_diffused_bgr.shape == bgr.shape:
            bgr = cv2.addWeighted(
                bgr, 1.0 - args.temporal_blend,
                last_diffused_bgr, args.temporal_blend,
                0.0,
            )
        last_diffused_bgr = bgr.copy() if args.temporal_blend > 0.0 else None

        now = time.time()
        H, W = bgr.shape[:2]

        if coin is not None:
            coin_size = max(16, int(W * COIN_REL_SIZE))
            face = coin.render(coin_size, now)
            if face is not None:
                fw = face.shape[1]
                cx = (W - fw) // 2
                cy = H - coin_size - int(H * COIN_BOTTOM_MARGIN_REL)
                composite_bgra(bgr, face, cx, cy)

        with transcripts_lock:
            draw_transcripts(bgr, transcripts, now)

        cv2.imshow(WINDOW, bgr)
        target_ms = max(1, int(batch_seconds * 1000 / args.batch))
        key = cv2.waitKey(target_ms) & 0xFF
        frames_displayed += 1

        if key == ord("q"):
            quit_requested = True
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
        elif key == ord("p"):
            new_prompt = input("new prompt: ").strip()
            if new_prompt:
                prompt = new_prompt
                worker.update_prompt(prompt)
                print(f"prompt: {prompt!r}", flush=True)
        elif key == ord("s"):
            seed += 1
            worker.update_seed(seed)
            print(f"seed: {seed}", flush=True)

        if now - last_report >= 2.0:
            print(f"~{frames_displayed / (now - last_report):.1f} fps", flush=True)
            frames_displayed = 0
            last_report = now

    if voice is not None:
        voice.close()
    worker.close()
    capture.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
