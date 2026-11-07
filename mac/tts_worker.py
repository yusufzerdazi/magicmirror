"""Kokoro text-to-speech, as a long-running helper process.

Runs in its own venv (.venv-tts, Python 3.10+, which kokoro-onnx needs) so the
mirror's own environment is untouched. Protocol, one request at a time:

    stdin   {"text": "...", "voice": "af_heart", "lang": "en-us", "speed": 1.0}\n
    stdout  {"ok": true, "rate": 24000, "bytes": N}\n  followed by N bytes of
            little-endian int16 mono PCM
            {"ok": false, "error": "..."}\n  on failure

The first line written is {"ready": true, "voices": [...]} once the model is
loaded. Nothing is written to disk: the audio goes straight back down the pipe.
"""

import json
import os
import sys

# Libraries that print would corrupt the binary protocol, so keep a private
# handle on the real stdout and send everything else to stderr.
_out = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr


def _send(header, payload=b""):
    _out.write(json.dumps(header).encode() + b"\n")
    if payload:
        _out.write(payload)
    _out.flush()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "models", "kokoro")
    import numpy as np
    from kokoro_onnx import Kokoro

    kokoro = Kokoro(os.path.join(model_dir, "kokoro-v1.0.onnx"),
                    os.path.join(model_dir, "voices-v1.0.bin"))
    # A first synthesis warms the runtime so the first real reply is quick.
    kokoro.create("Tuning in.", voice="af_heart", speed=1.0, lang="en-us")
    _send({"ready": True, "voices": sorted(kokoro.get_voices())})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            samples, rate = kokoro.create(
                req["text"], voice=req.get("voice", "af_heart"),
                speed=float(req.get("speed", 1.0)), lang=req.get("lang", "en-us"))
            pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            _send({"ok": True, "rate": int(rate), "bytes": len(pcm)}, pcm)
        except Exception as e:      # keep serving; the mirror falls back to `say`
            _send({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
