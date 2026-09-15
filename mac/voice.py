"""Microphone capture and transcription, matching the hosted pipeline.

The frontend records 10-second chunks with MediaRecorder and posts each one to
`/transcribe`; `backend/speech_processor.py` runs Whisper "base" on it, keeps
only segments with `avg_logprob > -1`, rejects repetitive output, and the route
then runs the text through moderation. This does all of that in-process.

Two transcription backends are supported:

* ``faster`` — faster-whisper on the CPU, the same library, model size, compute
  type and beam size the backend uses. This is the default.
* ``mlx`` — mlx-whisper on the GPU. Faster, but it has no beam search decoder,
  so it decodes greedily and will not always agree with the hosted build.
"""

import queue
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
# WarpPage's AUDIO_INTERVAL.
CHUNK_SECONDS = 10.0

# SpeechProcessor's decode settings.
# The backend runs "base", which is the weakest of these by some distance once
# there is any room noise; `distil-large-v3` is the default here instead.
FASTER_MODELS = {
    "base": "base",                  # the hosted build's model
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-small": "Systran/faster-distil-whisper-small.en",
}
FASTER_MODEL = "distil-large-v3"
FASTER_COMPUTE_TYPE = "int8"
MLX_MODELS = {
    "base": "mlx-community/whisper-base-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
MLX_MODEL = "large-v3-turbo"
BEAM_SIZE = 5
LANGUAGE = "en"
TEMPERATURE = 0.0
MIN_SEGMENT_LOGPROB = -1.0

BACKENDS = ("auto", "faster", "mlx")


class SpeechGate:
    """Drop chunks with no speech in them, before transcription.

    Whisper invents confident text out of room tone — "Thank you.", "I'm sorry,
    I'm sorry, ..." — and the hosted build only catches that when the invention
    happens to be repetitive. Gating on Silero VAD stops it at the source, and
    doing it here rather than inside one backend means the GPU path is covered
    too (mlx-whisper has no VAD of its own).
    """

    def __init__(self):
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        self._get_speech_timestamps = get_speech_timestamps
        self._options = VadOptions()

    def speech_only(self, audio):
        """Return `audio` trimmed to its speech, or None if there is none."""
        spans = self._get_speech_timestamps(audio, self._options,
                                            sampling_rate=SAMPLE_RATE)
        if not spans:
            return None
        start = max(0, spans[0]["start"])
        end = min(len(audio), spans[-1]["end"])
        return audio[start:end] if end > start else None


def is_repetitive(text, last_transcription):
    """Port of SpeechProcessor.is_repetitive."""
    words = text.lower().split()
    if len(words) < 4:  # Too short to be repetitive
        return False
    if last_transcription and text.lower() == last_transcription.lower():
        return True
    for i in range(1, len(words) // 2):
        pattern = words[:i]
        repetitions = [words[j:j + i] for j in range(0, len(words), i)]
        if len(repetitions) >= 3 and all(r == pattern for r in repetitions[:-1]):
            return True
    return False


class _FasterWhisper:
    """faster-whisper on the CPU — the backend's configuration exactly."""

    name = "faster-whisper"

    def __init__(self, model=FASTER_MODEL, vad=False,
                 compute_type=FASTER_COMPUTE_TYPE):
        from faster_whisper import WhisperModel
        repo = FASTER_MODELS.get(model, model)
        self.model = WhisperModel(
            model_size_or_path=repo,
            device="cpu",
            compute_type=compute_type,
        )
        self.vad = vad
        self.description = (f"faster-whisper {model} (cpu/{compute_type}, "
                            f"beam {BEAM_SIZE}{', vad' if vad else ''})")

    def segments(self, audio):
        segments, _info = self.model.transcribe(
            audio, beam_size=BEAM_SIZE, language=LANGUAGE, temperature=TEMPERATURE,
            vad_filter=self.vad,
        )
        return [(s.text, s.avg_logprob) for s in segments]


class _MlxWhisper:
    """mlx-whisper on the GPU; greedy, because MLX has no beam search decoder."""

    name = "mlx-whisper"

    def __init__(self, model=MLX_MODEL, vad=False, compute_type=None):
        import mlx_whisper
        self._transcribe = mlx_whisper.transcribe
        self.model = MLX_MODELS.get(model, model)
        self.description = f"mlx-whisper {model} (gpu, greedy)"

    def segments(self, audio):
        result = self._transcribe(
            audio, path_or_hf_repo=self.model,
            language=LANGUAGE, temperature=TEMPERATURE,
        )
        return [(s.get("text", ""), s.get("avg_logprob"))
                for s in (result.get("segments") or [])]


def build_transcriber(backend="auto", model=None, vad=False,
                      compute_type=FASTER_COMPUTE_TYPE):
    """Return a transcriber, preferring the GPU one.

    `auto` picks mlx-whisper large-v3-turbo, which is both the most accurate of
    these and the fastest, and costs the diffusion loop about 2% of its frame
    rate because it runs for a third of a second once per chunk. faster-whisper
    on the CPU is the fallback.
    """
    order = ("mlx", "faster") if backend == "auto" else (backend,)
    errors = []
    for choice in order:
        try:
            if choice == "faster":
                return _FasterWhisper(model or FASTER_MODEL, vad=vad,
                                      compute_type=compute_type)
            return _MlxWhisper(model or MLX_MODEL, vad=vad)
        except Exception as e:
            errors.append(f"{choice}: {e}")
    raise RuntimeError("; ".join(errors))


class VoiceListener:
    """Continuously records fixed-length chunks and transcribes them off-thread."""

    def __init__(self, on_transcript, backend="auto", model=None,
                 chunk_seconds=CHUNK_SECONDS, safety_checker=None,
                 playback=False, input_device=None, gate=True,
                 compute_type=FASTER_COMPUTE_TYPE, mute_while=None):
        self.on_transcript = on_transcript
        self.backend = backend
        self.model = model
        self.chunk_seconds = chunk_seconds
        self.gate = gate
        # Called before audio is kept. While the mirror is talking the
        # microphone hears it, and without this it transcribes its own reply and
        # answers itself.
        self.mute_while = mute_while
        self._gate = None
        self.compute_type = compute_type
        self.safety_checker = safety_checker
        self.playback = playback
        self.input_device = input_device
        self.stop_flag = threading.Event()
        self.last_transcription = None
        self._transcriber = None
        self._chunks = queue.Queue(maxsize=2)
        self._buffer = []
        self._buffer_samples = 0
        self._buffer_lock = threading.Lock()
        self._stream = None
        self._worker = None

    def try_start(self):
        try:
            import sounddevice as sd
        except ImportError as e:
            print(f"voice disabled — install sounddevice to enable ({e})", flush=True)
            return False

        if self.gate:
            try:
                self._gate = SpeechGate()
            except Exception as e:
                print(f"voice: speech gate unavailable ({e}); continuing without it",
                      flush=True)
        try:
            self._transcriber = build_transcriber(
                self.backend, self.model, False, self.compute_type)
        except Exception as e:
            print(f"voice disabled — no transcription backend available ({e})",
                  flush=True)
            return False

        chunk_samples = int(SAMPLE_RATE * self.chunk_seconds)

        def callback(indata, frames, time_info, status):
            if status:
                print(f"voice input status: {status}", flush=True)
            if self.mute_while is not None and self.mute_while():
                with self._buffer_lock:      # drop our own voice entirely
                    self._buffer = []
                    self._buffer_samples = 0
                return
            with self._buffer_lock:
                self._buffer.append(indata[:, 0].copy())
                self._buffer_samples += frames
                if self._buffer_samples < chunk_samples:
                    return
                audio = np.concatenate(self._buffer)
                self._buffer = []
                self._buffer_samples = 0
            try:
                self._chunks.put_nowait(audio)
            except queue.Full:
                pass  # transcription is behind; drop rather than fall further behind

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=int(SAMPLE_RATE * 0.25), device=self.input_device,
                callback=callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"voice disabled — could not open the microphone ({e})", flush=True)
            return False

        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        print(f"voice: ready ({self._transcriber.description}"
              f"{', speech-gated' if self._gate is not None else ''}, "
              f"chunk={self.chunk_seconds:g}s)", flush=True)
        return True

    def _loop(self):
        while not self.stop_flag.is_set():
            try:
                audio = self._chunks.get(timeout=0.25)
            except queue.Empty:
                continue
            if self.stop_flag.is_set():
                return
            if self.playback:
                self._play(audio)
            try:
                text = self._process(audio)
            except Exception as e:
                print(f"voice error: {e}", flush=True)
                time.sleep(0.5)
                continue
            if text:
                self.on_transcript(text)

    def _play(self, audio):
        """The frontend plays each recorded chunk back to the room; opt-in here."""
        try:
            import sounddevice as sd
            sd.play(audio, SAMPLE_RATE)
        except Exception as e:
            print(f"voice playback error: {e}", flush=True)

    def _process(self, audio):
        audio = audio.astype(np.float32)
        if self._gate is not None:
            audio = self._gate.speech_only(audio)
            if audio is None:
                return None
        segments = self._transcriber.segments(audio)

        # Keep only confident segments, exactly as SpeechProcessor does.
        parts = []
        confidence_total = 0.0
        segment_count = 0
        for text, avg_logprob in segments:
            if avg_logprob is not None and avg_logprob <= MIN_SEGMENT_LOGPROB:
                continue
            parts.append(text)
            if avg_logprob is not None:
                confidence_total += float(np.exp(avg_logprob))
            segment_count += 1

        if not segment_count:
            return None

        text = " ".join(p.strip() for p in parts if p.strip()).strip()
        if not text:
            return None
        confidence = confidence_total / segment_count
        print(f"voice: {text!r} (confidence: {confidence:.2f})", flush=True)

        if is_repetitive(text, self.last_transcription):
            print("voice: rejected, repetitive text detected", flush=True)
            return None
        self.last_transcription = text

        if self.safety_checker is not None:
            is_safe, message = self.safety_checker.check_transcription(text)
            if not is_safe:
                print(f"voice: rejected, {message}", flush=True)
                return None

        return text

    def close(self):
        self.stop_flag.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        if self._worker is not None:
            self._worker.join(timeout=2.0)
