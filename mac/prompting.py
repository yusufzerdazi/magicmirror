"""Prompt state and conditioning, matching the backend's settings API.

The Windows/web backend keeps two prompts and a blend factor (`/prompt/`,
`/secondprompt/`, `/blend/` in `backend/settings_api.py`) and hands compel a
string of the form `("a", "b").blend(0.40, 0.60)`, which
`DiffusionProcessor.meta_embed_prompt` splits back apart and lerps. This does
the same thing locally, using compel when it is installed and diffusers'
own encoder otherwise.
"""

import re
import threading

import torch

BLEND_PATTERN = re.compile(r'\("(.*?)"\s*,\s*"(.*?)"\)\.blend\((.*?),(.*?)\)')


class PromptState:
    """Two prompts and a blend factor, composed the way the backend composes them."""

    def __init__(self, prompt, second_prompt, blend=0.0):
        self._lock = threading.Lock()
        self._prompt_0 = prompt
        self._prompt_1 = second_prompt
        self._blend = blend

    def _compose_locked(self):
        if self._blend <= 0.0:
            return self._prompt_0
        if self._blend >= 1.0:
            return self._prompt_1
        t = self._blend
        return f'("{self._prompt_0}", "{self._prompt_1}").blend({1 - t:.2f}, {t:.2f})'

    def composed(self):
        with self._lock:
            return self._compose_locked()

    def set_prompt(self, prompt):
        with self._lock:
            self._prompt_0 = prompt
            return self._compose_locked()

    def set_second_prompt(self, prompt):
        with self._lock:
            self._prompt_1 = prompt
            return self._compose_locked()

    def set_blend(self, blend):
        with self._lock:
            self._blend = min(max(float(blend), 0.0), 1.0)
            return self._compose_locked()

    def describe(self):
        with self._lock:
            return self._prompt_0, self._prompt_1, self._blend


class PromptEncoder:
    """Turns a composed prompt into SDXL conditioning, cached per batch size."""

    def __init__(self, pipe, device, use_compel=True):
        self.pipe = pipe
        self.device = device
        # SDXL carries two text encoders and needs a pooled embedding alongside
        # the sequence one; SD 1.x/2.x pipelines have neither.
        self.is_sdxl = getattr(pipe, "text_encoder_2", None) is not None
        self.compel = None
        if use_compel:
            self.compel = self._try_build_compel(pipe, self.is_sdxl)
        self._cache = {}

    @staticmethod
    def _try_build_compel(pipe, is_sdxl):
        try:
            from compel import Compel, ReturnedEmbeddingsType
        except ImportError:
            return None
        try:
            if is_sdxl:
                return Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=(
                        ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
                    ),
                    requires_pooled=[False, True],
                )
            return Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        except Exception as e:  # pragma: no cover - compel is optional
            print(f"compel unavailable ({e}); using diffusers prompt encoding", flush=True)
            return None

    @property
    def backend_name(self):
        return "compel" if self.compel is not None else "diffusers"

    def _embed_single(self, prompt):
        """Embed one plain prompt at batch size 1.

        Always encoded at batch size 1 and expanded later: on MPS,
        `encode_prompt(prompt=[p] * N)` returns a different embedding for batch
        position 0 than for 1..N-1, which makes the first frame of every batch
        look different. See diag_batch.py.
        """
        if self.compel is not None:
            with torch.no_grad():
                result = self.compel(prompt)
            if self.is_sdxl:
                embeds, pooled = result
                return embeds.to(self.device), pooled.to(self.device)
            return result.to(self.device), None
        with torch.no_grad():
            if self.is_sdxl:
                embeds, _, pooled, _ = self.pipe.encode_prompt(
                    prompt=prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                return embeds, pooled
            embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        return embeds, None

    def _embed_composed(self, prompt):
        """Embed a prompt, interpolating if it is a `.blend(...)` expression."""
        match = BLEND_PATTERN.search(prompt)
        if not match:
            return self._embed_single(prompt)
        a, b, t1, t2 = match.groups()
        t1, t2 = float(t1), float(t2)
        cond_a, pooled_a = self._embed_single(a)
        cond_b, pooled_b = self._embed_single(b)
        pooled = None if pooled_a is None else pooled_a * t1 + pooled_b * t2
        return cond_a * t1 + cond_b * t2, pooled

    def encode(self, prompt, batch_size):
        key = (prompt, batch_size)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        embeds, pooled = self._embed_composed(prompt)
        embeds = embeds.expand(batch_size, -1, -1)
        if pooled is not None:
            pooled = pooled.expand(batch_size, -1)
        if len(self._cache) > 32:  # matches the backend's FixedSizeDict(32)
            self._cache.clear()
        self._cache[key] = (embeds, pooled)
        return embeds, pooled
