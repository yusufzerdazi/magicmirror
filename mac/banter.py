"""Cheeky spoken replies to whatever the mirror just heard.

A small local language model writes one short line back, and Kokoro (a small
neural voice, see tts.py) speaks it — or macOS `say` if Kokoro is unavailable.
Both run on the machine — nothing leaves it, and there is no network dependency
at a venue.

Output goes to a chosen device — the handset's earpiece rather than the room —
so the microphone in the same handset does not hear it and nothing needs muting.
`is_speaking()` is still exposed for setups where the speaker is open to the
room, where the mirror would otherwise transcribe its own reply and answer
itself; wire it to the voice listener's `mute_while` in that case.
"""

import queue
import random
import re
import subprocess
import threading
import time

# ~1B parameters at 4 bits, which is plenty for one sarcastic sentence and
# leaves the GPU to the diffusion loop.
DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

SYSTEM_PROMPT = (
    "You are the voice of a mischievous magic mirror at a party called Mischief. "
    "Someone just spoke to you. Reply with ONE short sentence, at most 12 words. "
    "Be cheeky, playful and a little smug — never cruel, never rude about "
    "anyone's appearance, never explicit. Do not use emoji, quotation marks or "
    "stage directions. Just the sentence."
)

# Used when the model is unavailable or too slow, and as the occasional
# interjection so the mirror still has a personality without one.
CANNED = (
    "Bold words for someone standing in front of a mirror.",
    "I heard you. I'm choosing to ignore you.",
    "That's the third time someone's asked for that tonight.",
    "Careful what you wish for, I take requests literally.",
    "I could do that. I might not.",
    "Interesting. Wrong, but interesting.",
    "Consider it done. Probably.",
    "Everyone says that. Nobody means it.",
)

# macOS voices. The novelty ones suit a mirror that is enjoying itself.
VOICES = ("Bad News", "Bubbles", "Boing", "Wobble", "Daniel")
DEFAULT_VOICE = "Daniel"
SPEECH_RATE = 180


def _clean(text):
    """One tidy sentence, whatever the model actually produced."""
    text = text.strip().strip('"').strip()
    text = re.sub(r"^\s*(mirror|assistant)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"[\*_`#]", "", text)
    text = re.sub(r"\s+", " ", text)
    # First sentence only; small models like to keep going.
    parts = re.split(r"(?<=[.!?])\s+", text)
    if parts:
        text = parts[0]
    words = text.split()
    if len(words) > 16:
        text = " ".join(words[:16]).rstrip(",;:") + "."
    return text.strip()


class Banter:
    """Writes and speaks a reply, one at a time, off the render thread."""

    def __init__(self, model=DEFAULT_MODEL, voice=DEFAULT_VOICE, rate=SPEECH_RATE,
                 use_llm=True, on_reply=None, reply_chance=1.0, output_device=None,
                 tts=None, output_name=None, persona=None, tts_voice=None):
        self.model_name = model
        self.voice = voice
        self.rate = rate
        self.use_llm = use_llm
        self.on_reply = on_reply
        self.reply_chance = reply_chance
        # `say -a` id of the output to speak through — the handset, normally.
        # None leaves it on the system default.
        self.output_device = output_device
        # Natural speech: a tts.KokoroVoice, played to the sounddevice output
        # that matches `output_name`. persona() -> (system prompt, Kokoro
        # voice) lets each radio station have its own host.
        self.tts = tts
        self.output_name = output_name
        self.persona = persona
        self.tts_voice = tts_voice
        self._play_index = None
        self._play_index_for = object()
        self._model = None
        self._tokenizer = None
        self._queue = queue.Queue(maxsize=1)
        self._speaking = threading.Event()
        self._stop = threading.Event()
        self._proc = None
        self._thread = None

    def is_speaking(self):
        return self._speaking.is_set()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.use_llm:
            threading.Thread(target=self._warm, daemon=True).start()
        return True

    def _warm(self):
        try:
            self._ensure_model()
            print(f"banter: {self.model_name} ready", flush=True)
        except Exception as e:
            print(f"banter: language model unavailable ({e}); using canned lines",
                  flush=True)
            self.use_llm = False

    def _ensure_model(self):
        if self._model is not None:
            return
        from mlx_lm import load

        self._model, self._tokenizer = load(self.model_name)

    def hear(self, text):
        """Offer something that was heard; ignored if a reply is in flight."""
        if not text or self._speaking.is_set():
            return False
        if random.random() > self.reply_chance:
            return False
        try:
            self._queue.put_nowait(text)
            return True
        except queue.Full:
            return False

    def _persona(self):
        system, voice = SYSTEM_PROMPT, self.tts_voice
        if self.persona is not None:
            try:
                system, host = self.persona()
                voice = voice or host
            except Exception as e:
                print(f"banter: persona failed ({e})", flush=True)
        return system, voice

    def _compose(self, heard, system=None):
        if self.use_llm and self._model is not None:
            try:
                from mlx_lm import generate

                messages = [{"role": "system", "content": system or SYSTEM_PROMPT},
                            {"role": "user", "content": heard}]
                prompt = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False)
                raw = generate(self._model, self._tokenizer, prompt=prompt,
                               max_tokens=40, verbose=False)
                line = _clean(raw)
                if line:
                    return line
            except Exception as e:
                print(f"banter: generation failed ({e})", flush=True)
        return random.choice(CANNED)

    def _speak_natural(self, line, voice):
        """Kokoro through sounddevice; False if that was not possible."""
        if self.tts is None or not self.tts.available():
            return False
        import tts as tts_module

        got = self.tts.synthesise(line, voice or tts_module.DEFAULT_VOICE)
        if got is None:
            return False
        if self._play_index_for != self.output_name:
            self._play_index_for = self.output_name
            self._play_index = tts_module.output_index(self.output_name)
        try:
            tts_module.play(got[0], got[1], device=self._play_index, gain=self.tts.gain)
            return True
        except Exception as e:
            print(f"banter: playback failed ({e})", flush=True)
            return False

    def _speak(self, line, voice=None):
        self._speaking.set()
        try:
            if self._speak_natural(line, voice):
                return
            command = ["say", "-v", self.voice, "-r", str(self.rate)]
            if self.output_device is not None:
                command += ["-a", str(self.output_device)]
            command.append(line)
            self._proc = subprocess.Popen(command)
            self._proc.wait()
        except Exception as e:
            print(f"banter: speech failed ({e})", flush=True)
        finally:
            self._proc = None
            # The microphone keeps hearing the tail of the speech for a moment
            # after the process exits.
            time.sleep(0.6)
            self._speaking.clear()

    def _loop(self):
        while not self._stop.is_set():
            try:
                heard = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            system, voice = self._persona()
            line = self._compose(heard, system)
            if not line:
                continue
            print(f"banter: {line!r}" + (f" ({voice})" if voice else ""), flush=True)
            if self.on_reply is not None:
                self.on_reply(line)
            self._speak(line, voice)

    def close(self):
        self._stop.set()
        if self.tts is not None:
            self.tts.close()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
