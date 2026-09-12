"""Transcript moderation, ported from `backend/safety_checker.py`.

The hosted `/transcribe` endpoint runs every transcription through a hate-speech
classifier and drops the text if it comes back unsafe, so the local app does the
same before a transcript reaches the screen or the prompt.
"""

import re
from functools import lru_cache

MODEL_NAME = "facebook/roberta-hate-speech-dynabench-r4-target"

# `backend/settings_api.py` carries this list alongside the classifier. It is
# not wired into any route there, so it is opt-in here (--banned-words).
BANNED_WORDS = [
    "Blood", "Bloodbath", "Crucifixion", "Bloody", "Flesh", "Bruises", "Car crash",
    "Corpse", "Crucified", "Cutting", "Decapitate", "Infested", "Gruesome",
    "Infected", "Sadist", "Slaughter", "Teratoma", "Tryphophobia", "Wound",
    "Cronenberg", "Khorne", "Cannibal", "Cannibalism", "Visceral", "Guts",
    "Bloodshot", "Gory", "Killing", "Surgery", "Vivisection", "Massacre",
    "Hemoglobin", "Suicide",
    "Drugs", "Cocaine", "Heroin", "Meth", "Crack",
    "no clothes", "Speedo", "au naturale", "no shirt", "bare chest", "nude",
    "barely dressed", "bra", "risqué", "scantily", "clad", "cleavage", "stripped",
    "full frontal unclothed", "invisible clothes", "wearing nothing", "naked",
    "without clothes on", "negligee", "zero clothes",
    "ahegao", "pinup", "ballgag", "Playboy", "Bimbo", "bodily fluids", "boudoir",
    "rule34", "brothel", "seducing", "dominatrix", "seductive", "erotic", "fuck",
    "sensual", "Hardcore", "sexy", "Hentai", "Shag", "horny", "shibari", "incest",
    "Smut", "jav", "succubus", "thot", "kinbaku", "legs spread", "twerk",
    "making love", "voluptuous", "naughty", "wincest", "orgy", "Sultry", "XXX",
    "Bondage", "Bdsm", "Dog collar", "Slavegirl",
    "Torture", "Disturbing", "Farts", "Fart", "Poop", "Warts", "Xi Jinping",
    "Shit", "Errect", "Big Black", "Brown pudding", "Bunghole", "Vomit", "Sperm",
    "Sensored", "Censored", "Silenced", "Deepfake", "Inappropriate", "Pus",
    "Waifu", "mp5", "1488",
]


class SafetyChecker:
    """Hate-speech classifier with the same safe/unsafe contract as the backend."""

    def __init__(self, use_banned_words=False, device=-1):
        self.use_banned_words = use_banned_words
        self.device = device
        self._classifier = None
        # Whole words/phrases only, so "octopus" is not a hit for "pus".
        self._banned = re.compile(
            r"\b(?:%s)\b" % "|".join(re.escape(w.lower()) for w in BANNED_WORDS),
            re.IGNORECASE,
        )

    def _ensure_model(self):
        if self._classifier is not None:
            return self._classifier
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        # Kept off the MPS device on purpose: it is a tiny model and the
        # diffusion loop needs the GPU.
        self._classifier = pipeline(
            "text-classification", model=model, tokenizer=tokenizer, device=self.device
        )
        return self._classifier

    def warm_up(self):
        """Load the model up front so the first transcript is not delayed."""
        try:
            self._ensure_model()
            return True
        except Exception as e:
            print(f"moderation unavailable ({e}); transcripts will not be checked",
                  flush=True)
            return False

    def _banned_hit(self, text):
        match = self._banned.search(text)
        return match.group(0) if match else None

    @lru_cache(maxsize=1000)
    def _classify(self, text):
        classifier = self._ensure_model()
        result = classifier(text)[0]
        return result["label"] == "nothate", float(result["score"])

    def __call__(self, text):
        """Return ("safe"|"unsafe", detail), mirroring SafetyChecker.__call__."""
        if not text:
            return "safe", "empty"
        if self.use_banned_words:
            hit = self._banned_hit(text)
            if hit:
                return "unsafe", f"Contains banned word: {hit!r}"
        try:
            is_safe, confidence = self._classify(text)
        except Exception as e:
            print(f"Error in local model classification: {e}", flush=True)
            return "safe", "error"
        if is_safe:
            return "safe", f"Content is appropriate (confidence: {confidence:.2f})"
        return "unsafe", f"Content may be inappropriate (confidence: {confidence:.2f})"

    def check_transcription(self, text):
        """Return (is_safe, message), mirroring SafetyChecker.check_transcription."""
        result, details = self(text)
        if result == "unsafe":
            return False, f"Content moderation check failed: {details}"
        return True, "Content is appropriate"
