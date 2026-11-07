"""Natural speech for the mirror's replies: Kokoro, played into the handset.

Kokoro-82M is a small neural text-to-speech model that sounds like a person
rather than a speech synthesiser, and runs faster than real time on the CPU, so
it leaves the GPU to the picture. It needs Python 3.10+, so it runs as a helper
process (tts_worker.py) in its own venv; this side sends it a line of text and
plays the audio that comes back through the chosen output — the handset's
earpiece — with sounddevice.

Set up once (start.sh does this check for you):

    /opt/homebrew/bin/python3.13 -m venv .venv-tts
    .venv-tts/bin/pip install kokoro-onnx
    .venv/bin/python download_models.py --tts

If any of that is missing, or the helper dies, `available()` is False and the
mirror carries on with macOS `say`.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TTS_PYTHON = HERE / ".venv-tts" / "bin" / "python"
MODEL_DIR = HERE / "models" / "kokoro"
MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

DEFAULT_VOICE = "af_heart"


def lang_for(voice):
    """Kokoro voices are prefixed a (American) or b (British) English."""
    return "en-gb" if voice.startswith("b") else "en-us"


def installed():
    return TTS_PYTHON.exists() and all((MODEL_DIR / f).exists() for f in MODEL_FILES)


class KokoroVoice:
    """The helper process, started in the background so startup never waits."""

    def __init__(self, speed=1.0, gain=0.8):
        self.speed = speed
        self.gain = gain
        self.voices = ()
        self._proc = None
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        if not installed():
            print("tts: Kokoro is not installed; replies use macOS `say`", flush=True)
            self._failed.set()
            return False
        threading.Thread(target=self._launch, daemon=True).start()
        return True

    def _launch(self):
        try:
            self._proc = subprocess.Popen(
                [str(TTS_PYTHON), str(HERE / "tts_worker.py"), str(MODEL_DIR)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
            header = json.loads(self._proc.stdout.readline() or b"{}")
            if not header.get("ready"):
                raise RuntimeError("helper did not start")
            self.voices = tuple(header.get("voices", ()))
            self._ready.set()
            print(f"tts: Kokoro ready ({len(self.voices)} voices)", flush=True)
        except Exception as e:
            print(f"tts: Kokoro unavailable ({e}); replies use macOS `say`", flush=True)
            self._failed.set()

    def available(self):
        return self._ready.is_set() and not self._failed.is_set()

    def synthesise(self, text, voice=DEFAULT_VOICE):
        """Mono float32 samples and their rate, or None."""
        if not self.available():
            return None
        if self.voices and voice not in self.voices:
            voice = DEFAULT_VOICE
        request = {"text": text, "voice": voice, "lang": lang_for(voice),
                   "speed": self.speed}
        with self._lock:
            try:
                self._proc.stdin.write((json.dumps(request) + "\n").encode())
                self._proc.stdin.flush()
                header = json.loads(self._proc.stdout.readline() or b"{}")
                if not header.get("ok"):
                    print(f"tts: {header.get('error', 'helper went away')}", flush=True)
                    if not header:
                        self._failed.set()
                    return None
                raw = self._read_exactly(header["bytes"])
            except Exception as e:
                print(f"tts: helper failed ({e}); falling back to `say`", flush=True)
                self._failed.set()
                return None
        samples = np.frombuffer(raw, "<i2").astype(np.float32) / 32767.0
        return samples, int(header["rate"])

    def _read_exactly(self, n):
        chunks, got = [], 0
        while got < n:
            chunk = self._proc.stdout.read(n - got)
            if not chunk:
                raise RuntimeError("helper closed the pipe")
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
            except Exception:
                pass


def output_index(name):
    """The sounddevice index of the output `say -a` calls `name`, or None."""
    try:
        import sounddevice as sd
        outputs = [(i, d) for i, d in enumerate(sd.query_devices())
                   if d.get("max_output_channels", 0) > 0]
    except Exception:
        return None
    if not name:
        return None
    for i, d in outputs:
        if d["name"] == name:
            return i
    low = name.lower()
    for i, d in outputs:
        if low in d["name"].lower() or d["name"].lower() in low:
            return i
    return None


def play(samples, rate, device=None, gain=0.8):
    """Play mono float32 audio and block until it has finished."""
    import sounddevice as sd

    info = sd.query_devices(device, "output")
    out_rate = int(info.get("default_samplerate") or rate)
    audio = samples.astype(np.float32)
    if out_rate != rate and audio.size:
        n = int(round(audio.size * out_rate / rate))
        audio = np.interp(np.linspace(0, audio.size - 1, n),
                          np.arange(audio.size), audio).astype(np.float32)
    # Even level whatever the voice, kept gentle: it is an earpiece.
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 1e-4:
        audio *= gain / peak
    fade = min(audio.size // 2, int(out_rate * 0.008))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    # A short lead-in so the first syllable is not lost while the device wakes.
    audio = np.concatenate([np.zeros(int(out_rate * 0.12), np.float32), audio])
    channels = min(2, int(info.get("max_output_channels", 1)) or 1)
    if channels == 2:
        audio = np.repeat(audio[:, None], 2, axis=1)
    sd.play(audio, out_rate, device=device, blocking=True)
