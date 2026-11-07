"""DREAMWAVE — the London Decompression 2026 dressing of the mirror.

The event's theme: somewhere, in a reality that never fully happened, a signal
has been broadcasting all along, waiting for people willing to receive it.
"Tune in. The dream is real." The poster is a surreal paper collage — a pink
classical archway opening onto blue sky and clouds, coral and cream, gold
constellations on charcoal, vintage radios, a crow, an eye on a radio mast.

So the mirror becomes a radio receiver for dreams:

* It is always tuned to a **station**, and each station is a dream the people
  in front of it are redrawn into. Every so often it retunes: the needle sweeps
  across the dial, static rolls over the picture, and the prompt *morphs*
  between the two stations (the embeddings are interpolated, so you watch one
  dream turn into the next) rather than cutting.
* Someone new stepping into view tunes in a fresh station, so each arrival gets
  their own dream.
* Speaking into the handset goes **on air**: the dial shows LIVE, the words
  become the dream, and the mirror answers back in the earpiece like a
  late-night radio host from an alternate present.
* The picture is seen through the poster's archway, with the "tune in" ribbon
  across its top and the glowing coin — now a gold eye on a radio mast — at its
  foot.

Everything is drawn with OpenCV/PIL into cached layers. The render loop runs at
60Hz on the same machine as the diffusion model, so nothing full-screen is
alpha-blended per frame: static artwork is stored as a sparse list of the
pixels it covers, and the moving parts are small.
"""

import math
import random
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ASSETS_DIR = Path(__file__).parent / "assets"
DISPLAY_FONT = "Righteous-Regular.ttf"
DIAL_FONT = "VT323-Regular.ttf"

# --- palette, sampled from the 2026 poster (RGB) ------------------------------
INK = (27, 25, 30)
CHARCOAL = (43, 40, 44)
NIGHT = (24, 24, 46)
PLUM = (78, 44, 74)
CORAL = (217, 88, 92)        # the ribbon
SALMON = (240, 158, 140)     # the archway
BLUSH = (249, 196, 178)
CREAM = (247, 236, 216)
GOLD = (219, 176, 96)
SKY = (140, 196, 234)

TEXT_COLOURS = (CREAM, BLUSH, GOLD, SKY)


def bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def font_path(name=DISPLAY_FONT):
    path = ASSETS_DIR / name
    return str(path) if path.exists() else None


_FONTS = {}


def font(size, name=DISPLAY_FONT):
    key = (name, int(size))
    if key not in _FONTS:
        path = font_path(name)
        _FONTS[key] = (ImageFont.truetype(path, int(size)) if path
                       else ImageFont.load_default())
    return _FONTS[key]


# --- stations -----------------------------------------------------------------
# Every prompt ends in the shared style, so whatever the station, the picture
# belongs to the same poster.
STYLE = ("surreal dreamlike vintage paper collage, halftone print texture, "
         "art deco poster, rich colours")


class Station:
    def __init__(self, freq, name, subject, place):
        self.freq = freq
        self.name = name
        self.subject = subject
        self.place = place          # the world around you, in Dream World

    @property
    def prompt(self):
        return f"{self.subject}, {STYLE}"


# Each subject is a *figure*: the control image is a person's silhouette, and a
# prompt about a place rather than a being lets the model turn them into
# scenery — a mirror with nobody in it.
STATIONS = (
    Station(88.1, "Cloud Walkers",
            "full body portrait of a dreamer whose body is made of soft white clouds, "
            "vivid blue sky behind",
            "a kingdom of floating cloud islands and pink classical archways in a blue sky"),
    Station(90.7, "Radio Saints",
            "full body portrait of a radiant saint with a halo of radio waves and "
            "golden constellations on midnight blue, ornate tarot card",
            "a midnight cathedral of radio towers under golden constellations"),
    Station(93.3, "Moon Ballroom",
            "full body portrait of an elegant dancer in a silver sequined gown, giant "
            "moon and mirror ball, deep violet night",
            "a ballroom on the surface of the moon, giant mirror ball, violet night sky"),
    Station(95.9, "Deep Sea Static",
            "full body portrait of a glowing deep sea jellyfish person with "
            "bioluminescent tendrils, teal and turquoise underwater",
            "a glowing coral reef city deep under a turquoise sea"),
    Station(98.4, "Playa Mirage",
            "full body portrait of a desert wanderer in a fantastical fur coat and "
            "goggles, dust storm at golden hour, glowing art car",
            "an endless desert playa at golden hour with glowing art cars and a giant effigy"),
    Station(101.1, "The Hypnotist",
            "full body portrait of a mysterious hypnotist with an all-seeing eye, black "
            "and cream op-art spiral, red velvet curtains",
            "an infinite theatre of red velvet curtains and black and cream spirals"),
    Station(103.7, "Tin Radio Heart",
            "full body portrait of a vintage tin toy robot with a glowing radio dial for "
            "a heart, mint green and chrome",
            "a retro-futurist city of chrome radio towers and mint green skies"),
    Station(106.5, "Garden of Lost Keys",
            "full body portrait of a forest spirit made of flowers, antique keys and "
            "butterflies, lush green enchanted garden",
            "an enchanted garden of giant flowers, hanging antique keys and butterflies"),
)
DIAL_MIN, DIAL_MAX = 87.5, 108.0

# --- voice -> prompt ----------------------------------------------------------
_LEAD_INS = re.compile(
    r"^\s*(?:(?:hey|ok|okay|so|um+|uh+|please|can you|could you|i want you to|"
    r"i'd like (?:you )?to|i want to|i wanna|i would like to)[\s,]+)*"
    r"(?:turn (?:me|us|everyone|everybody|them|him|her) into|make (?:me|us|everyone|"
    r"everybody) (?:into )?|transform (?:me|us|everyone) into|"
    r"be(?:come)?|i am|i'm|we are|we're|show me|give me)?\s*",
    re.IGNORECASE)

# Decompression is 18+, but the mirror redraws whoever walks past it, and
# consent underpins the event. Words that would make it draw people naked or
# sexualised are heard and shown, never rendered onto anyone.
_NOT_ON_PEOPLE = re.compile(
    r"\b(nak(?:ed|ie)|nude|nudity|topless|bottomless|sex(?:y|ual|ually)?|porn\w*|"
    r"nsfw|boob\w*|tit(?:s|ties)?|breasts?|nipples?|penis|dick|cock|vagina|pussy|"
    r"genital\w*|lingerie|bikini|underwear|strip(?:per|ping)?|erotic\w*|horny|"
    r"orgasm\w*|fetish\w*|bdsm|bondage|blood(?:y)?|gore|corpse|dead body|"
    r"suicide|kill(?:ing|ed)?|murder\w*)\b",
    re.IGNORECASE)


# Small talk and questions are answered, not drawn: "hello?" should not turn
# the room into the word hello.
_NOT_A_DREAM = re.compile(
    r"^(?:hi|hello|hey|hiya|yo|ok|okay|yes|yeah|no|nope|thanks|thank you|testing|"
    r"test|bye|goodbye|what|why|how|who|where|when|is|are|can|could|does|do|"
    r"did|will|would|should)\b", re.IGNORECASE)
_TAIL = re.compile(r"[\s,]*(?:please|thanks|thank you|now|ok|okay)[\s.!?]*$",
                   re.IGNORECASE)


def dream_from_words(text):
    """What someone said -> the subject of a prompt, or None to leave it be."""
    if not text or _NOT_ON_PEOPLE.search(text):
        return None
    words = text.replace('"', "").replace("(", "").replace(")", "").strip()
    if _NOT_A_DREAM.match(words.strip(" .,!?")) and "into" not in words.lower():
        return None
    words = _LEAD_INS.sub("", words, count=1).strip(" .,!?;:")
    words = _TAIL.sub("", words).strip(" .,!?;:")
    if len(words) < 3:
        return None
    return words[:160]


def live_prompt(subject):
    return f"full body portrait, {subject}, {STYLE}"


# --- banter -------------------------------------------------------------------
BANTER_SYSTEM_PROMPT = (
    "You are the voice of a magic mirror at London Decompression, a Burning Man "
    "party whose theme is Dreamwave: a pirate radio station broadcasting from a "
    "dream that turned out to be real. Someone just spoke into your telephone "
    "handset and you are about to redraw them as what they said. Reply like a "
    "warm, mischievous late-night radio host from an alternate reality: ONE "
    "short sentence, at most 14 words. Playful, a little cryptic, never cruel, "
    "never about anyone's body or appearance, never explicit. No emoji, no "
    "quotation marks, no stage directions. Just the sentence."
)

BANTER_CANNED = (
    "Signal received. The dream is real, and so are you.",
    "Coming to you live from a reality that never quite happened.",
    "Hold still, I'm tuning you in.",
    "Lovely frequency. Stay on it.",
    "Broadcasting your dream to everyone on this wavelength.",
    "I heard that on a channel nobody else can reach.",
    "Request received. Dedicating this one to you.",
    "You're on air, darling. Try to look dreamy.",
    "Interesting choice. The static approves.",
    "Careful, I take requests very literally.",
)

IDLE_HEADLINE = "tune in"
IDLE_SUGGESTIONS = (
    "pick up the phone and tell me your dream",
    "pick up the phone and say what you want to become",
    "pick up the phone · try: turn us into clouds",
    "pick up the phone · whisper a frequency only you can hear",
    "pick up the phone · describe the world you'd rather live in",
    "the dream is real. so are you. pick up the phone",
)


# --- the tuner ----------------------------------------------------------------
def _ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class Tuner:
    """Which dream the mirror is receiving, and the retuning between them.

    Drives `PromptState`: while tuning, prompt 0 is where it came from, prompt 1
    where it is going, and the blend sweeps between them, so the picture morphs
    through the space between two prompts instead of cutting.
    """

    TUNE_SECONDS = 3.2          # one sweep of the needle
    LIVE_TUNE_SECONDS = 1.6     # quicker when someone has asked for something
    DWELL_SECONDS = 50.0        # on a station, with people about
    EMPTY_DWELL_SECONDS = 26.0  # an empty room channel-surfs to draw people in
    LIVE_SECONDS = 45.0         # how long a spoken dream stays on air
    MIN_BEFORE_ARRIVAL = 10.0   # an arrival does not retune a fresh station

    def __init__(self, prompt_state, stations=STATIONS, on_retune=None, rng=None,
                 clock=time.time):
        self.prompt_state = prompt_state
        self.stations = stations
        self.on_retune = on_retune      # called with a new seed per station
        self.rng = rng or random.Random()
        self.clock = clock
        self.lock = threading.Lock()
        self.index = self.rng.randrange(len(stations))
        self.freq = stations[self.index].freq
        self.from_freq = self.freq
        self.target_freq = self.freq
        self.from_prompt = stations[self.index].prompt
        self.target_prompt = self.from_prompt
        self.tune_started = None
        self.tune_seconds = self.TUNE_SECONDS
        self.since = clock()
        self.live_text = None
        self.live_until = 0.0
        self.paused = False         # a prompt typed by hand holds the dial
        self.tunes = 0              # station changes so far
        prompt_state.set_prompt(self.from_prompt)
        prompt_state.set_blend(0.0)

    # -- requests ------------------------------------------------------------
    def _begin(self, prompt, freq, seconds, now):
        self.from_prompt = self.prompt_state.describe()[0] if self.tune_started is None \
            else self._current_prompt_guess()
        self.from_freq = self.freq
        self.target_prompt = prompt
        self.target_freq = freq
        self.tune_started = now
        self.tune_seconds = seconds
        self.prompt_state.set_prompt(self.from_prompt)
        self.prompt_state.set_second_prompt(prompt)
        self.prompt_state.set_blend(0.0)
        if self.on_retune is not None:
            self.on_retune(self.rng.randrange(1, 10_000))

    def _current_prompt_guess(self):
        # Mid-sweep, whichever end is nearer stands in for "where we are".
        p0, p1, blend = self.prompt_state.describe()
        return p1 if blend >= 0.5 else p0

    def next_station(self, now=None, reason=""):
        now = self.clock() if now is None else now
        with self.lock:
            self.live_text = None
            choices = [i for i in range(len(self.stations)) if i != self.index]
            self.index = self.rng.choice(choices)
            st = self.stations[self.index]
            self._begin(st.prompt, st.freq, self.TUNE_SECONDS, now)
            self.since = now
            self.tunes += 1
        print(f"dreamwave: tuning to {st.freq:.1f} {st.name}"
              + (f" ({reason})" if reason else ""), flush=True)
        return st

    def go_live(self, subject, heard=None, now=None):
        """Someone spoke: their words go on air as the dream."""
        now = self.clock() if now is None else now
        with self.lock:
            self.live_text = heard or subject
            self.live_until = now + self.LIVE_SECONDS
            # A live request gets a frequency of its own, off the station marks.
            freq = round(self.rng.uniform(DIAL_MIN + 0.6, DIAL_MAX - 0.6), 1)
            self._begin(live_prompt(subject), freq, self.LIVE_TUNE_SECONDS, now)
            self.since = now
        print(f"dreamwave: ON AIR {subject!r}", flush=True)

    def hold(self, prompt):
        """A prompt typed by hand: show it, and stop the dial wandering off."""
        with self.lock:
            self.paused = True
            self.tune_started = None
            self.live_text = None
            self.prompt_state.set_prompt(prompt)
            self.prompt_state.set_blend(0.0)

    def resume(self):
        self.paused = False

    # -- per frame -------------------------------------------------------------
    def update(self, now, present=False, arrived=False):
        retune = None
        with self.lock:
            if self.tune_started is not None:
                k = (now - self.tune_started) / self.tune_seconds
                if k >= 1.0:
                    self.tune_started = None
                    self.freq = self.target_freq
                    self.prompt_state.set_prompt(self.target_prompt)
                    self.prompt_state.set_blend(0.0)
                else:
                    # Quantised so the encoder is not asked for a new blend on
                    # every display frame.
                    self.prompt_state.set_blend(round(_ease(k) * 10) / 10.0)
                    # The needle overshoots and settles, the way a hand on a
                    # real dial does.
                    e = _ease(min(1.0, k * 1.25))
                    wobble = math.sin(k * math.pi * 3.0) * (1.0 - k) * 0.6
                    self.freq = (self.from_freq + (self.target_freq - self.from_freq) * e
                                 + wobble)
                return
            if self.paused:
                return
            on_air = self.live_text is not None and now < self.live_until
            if self.live_text is not None and not on_air:
                retune = "end of transmission"
            elif not on_air:
                dwell = self.DWELL_SECONDS if present else self.EMPTY_DWELL_SECONDS
                if now - self.since >= dwell:
                    retune = "next station"
                elif arrived and now - self.since >= self.MIN_BEFORE_ARRIVAL:
                    retune = "someone new tuned in"
        if retune:
            self.next_station(now, retune)

    def static_level(self, now):
        """How much static is on the picture: a swell mid-sweep."""
        with self.lock:
            if self.tune_started is None:
                return 0.0
            k = (now - self.tune_started) / self.tune_seconds
        if not 0.0 <= k <= 1.0:
            return 0.0
        return 0.55 * math.sin(math.pi * k) ** 1.5

    def tuning(self):
        return self.tune_started is not None

    def station(self):
        return self.stations[self.index]

    def display(self, now):
        """(frequency, station name, on air?) for the dial."""
        with self.lock:
            on_air = self.live_text is not None and now < self.live_until
            if on_air:
                return self.freq, self.live_text, True
            if self.paused:
                return self.freq, "private channel", False
            return self.freq, self.stations[self.index].name, False


# --- presence -----------------------------------------------------------------
class Presence:
    """Is anyone in front of the mirror, and did someone just arrive?

    With a Kinect this is metric: anything inside the working volume, in the
    upper part of the frame (the floor creeps into the bottom of it). With only
    a camera it is motion, held for a while, since people standing still in
    front of a mirror are still there.
    """

    ARRIVAL_GAP = 4.0       # must have been empty this long for an arrival
    MOTION_HOLD = 8.0

    def __init__(self):
        self.present = False
        self.level = 0.0        # 0..1, smoothed, for the signal meter
        self.nearness = 0.0     # 0..1 with depth, how close the nearest person is
        self.centre_x = 0.5     # where people are, 0 left .. 1 right (mirrored)
        self.empty_since = time.time()
        self._prev = None
        self._last_motion = 0.0
        self._arrived = False

    def update_depth(self, depth_rgb, now):
        d = depth_rgb[:, :, 0] if depth_rgb.ndim == 3 else depth_rgb
        h = d.shape[0]
        top = d[: int(h * 0.68)]
        valid = top > 0
        fraction = float(valid.mean())
        raw = min(1.0, fraction / 0.18)
        if fraction > 0.01:
            self.nearness = float(top[valid].mean()) / 255.0
            cols = valid.mean(axis=0)
            self.centre_x = float((cols * np.arange(cols.size)).sum()
                                  / max(1e-6, cols.sum()) / cols.size)
        self._set(fraction > 0.035, raw, now)

    def update_camera(self, frame_bgr, now):
        small = cv2.cvtColor(cv2.resize(frame_bgr, (64, 36), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev is not None:
            motion = float(np.abs(small - self._prev).mean())
            if motion > 2.2:
                self._last_motion = now
            raw = min(1.0, motion / 10.0)
        else:
            raw = 0.0
        self._prev = small
        self._set(now - self._last_motion < self.MOTION_HOLD, max(raw, 0.25 if
                  now - self._last_motion < self.MOTION_HOLD else 0.0), now)

    def _set(self, present, raw, now):
        self.level += (raw - self.level) * 0.25
        if present and not self.present:
            if now - self.empty_since >= self.ARRIVAL_GAP:
                self._arrived = True
        if not present and self.present:
            self.empty_since = now
        self.present = present

    def take_arrival(self):
        arrived, self._arrived = self._arrived, False
        return arrived


# --- sparse layers --------------------------------------------------------------
class SparseLayer:
    """A full-screen RGBA overlay stored as only the pixels it covers.

    Thin linework over a 1080p screen covers a few percent of it; blending just
    those pixels costs a fraction of blending the whole frame.
    """

    def __init__(self, rgba):
        a = rgba[:, :, 3]
        ys, xs = np.nonzero(a)
        self.shape = rgba.shape[:2]
        self.ys, self.xs = ys, xs
        self.alpha = (a[ys, xs].astype(np.float32) / 255.0)[:, None]
        self.colour = rgba[ys, xs, :3][:, ::-1].astype(np.float32)   # -> BGR
        self.premul = self.colour * self.alpha
        self.keep = 1.0 - self.alpha

    def draw(self, canvas, opacity=1.0):
        if self.ys.size == 0:
            return
        under = canvas[self.ys, self.xs].astype(np.float32)
        if opacity >= 1.0:
            out = under * self.keep + self.premul
        else:
            a = self.alpha * opacity
            out = under * (1.0 - a) + self.colour * a
        canvas[self.ys, self.xs] = out.astype(np.uint8)


def blit_rgba(canvas, rgba, x, y, opacity=1.0):
    """Alpha-blend a small RGBA PIL/numpy image onto the BGR canvas."""
    if isinstance(rgba, Image.Image):
        rgba = np.asarray(rgba)
    H, W = canvas.shape[:2]
    h, w = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    piece = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = piece[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
    src = piece[:, :, 2::-1].astype(np.float32)
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (region + (src - region) * a).astype(np.uint8)


# --- the sky behind everything ---------------------------------------------------
class DuskSky:
    """Dusk gradient, gold constellations, and paper clouds drifting across.

    The gradient and constellations are drawn once at full resolution; each
    cloud is drawn once as a sprite and only moved, ten times a second, blended
    into just the rectangle it covers.
    """

    TICK_SECONDS = 0.1          # clouds drift a pixel or two a tick
    COUNT = 7

    def __init__(self, seed=11):
        self.rng = random.Random(seed)
        self.clouds = [self._new_cloud(initial=True) for _ in range(self.COUNT)]
        self._base = None
        self._base_size = None
        self._frame = None
        self._frame_size = None
        self._last = None
        self.version = 0            # bumps whenever the rendered sky changes

    def _new_cloud(self, initial=False):
        r = self.rng
        puffs = []
        n = r.randint(5, 8)
        for k in range(n):
            puffs.append((k / (n - 1) - 0.5 + r.uniform(-0.08, 0.08),
                          r.uniform(-0.18, 0.08) - 0.25 * math.sin(math.pi * k / (n - 1)),
                          r.uniform(0.18, 0.32)))
        return {
            "x": r.uniform(-0.1, 1.1) if initial else (-0.35 if r.random() < 0.5 else 1.35),
            "y": r.uniform(0.08, 0.95),
            "scale": r.uniform(0.1, 0.22),
            "speed": r.uniform(0.004, 0.012) * (1 if r.random() < 0.5 else -1),
            "puffs": puffs,
            "tint": r.choice((CREAM, BLUSH, CREAM, (232, 226, 240))),
            "alpha": r.uniform(0.82, 0.95),
            "sprite": None,
        }

    def _sprite(self, c, h):
        """A cloud cut from paper: drawn once at full size, then only moved."""
        if c["sprite"] is not None and c["sprite"][0] == h:
            return c["sprite"][1:]
        unit = h
        rad = [pr * c["scale"] * unit * 1.3 for _, _, pr in c["puffs"]]
        xs = [px * c["scale"] * 1.6 * unit for px, _, _ in c["puffs"]]
        ys = [py * c["scale"] * unit for _, py, _ in c["puffs"]]
        pad = 4
        x0 = int(min(x - r for x, r in zip(xs, rad))) - pad
        x1 = int(max(x + r for x, r in zip(xs, rad))) + pad
        y0 = int(min(y - r for y, r in zip(ys, rad))) - pad
        y1 = int(max(y + r for y, r in zip(ys, rad))) + pad
        mask = np.zeros((y1 - y0, x1 - x0), np.float32)
        for x, y, r in zip(xs, ys, rad):
            cv2.circle(mask, (int(x - x0), int(y - y0)), max(1, int(r)), 1.0, -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.2) * c["alpha"]
        rows = np.linspace(0.0, 1.0, mask.shape[0], dtype=np.float32)[:, None, None]
        shade = 1.0 - 0.24 * np.clip((rows - 0.35) / 0.65, 0, 1)
        colour = (np.array(bgr(c["tint"]), np.float32) * shade).astype(np.float32)
        colour = np.broadcast_to(colour, mask.shape + (3,)).astype(np.float32)
        # Offsets are relative to the cloud's anchor point.
        c["sprite"] = (h, colour, mask, x0, y0)
        return c["sprite"][1:]

    def _make_base(self, w, h):
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        stops = ((0.0, NIGHT), (0.55, PLUM), (0.85, (150, 74, 84)), (1.0, (196, 104, 94)))
        pos = np.array([p for p, _ in stops], np.float32)
        base = np.empty((h, w, 3), np.float32)
        for c in range(3):
            vals = np.array([col[2 - c] for _, col in stops], np.float32)
            base[:, :, c] = np.interp(ys, pos, vals)
        img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)[:, :, ::-1], "RGB")
        lines = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(lines)
        r = random.Random(3)
        gold = GOLD + (95,)
        unit = h
        # Astrolabe rings, low in each corner, as on the poster's charcoal panels.
        for cx, cy in ((0.08 * w, 0.82 * h), (w - 0.08 * w, 0.2 * h)):
            for k, rad in enumerate((0.16, 0.22, 0.3, 0.36)):
                rr = rad * unit
                d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=gold,
                          width=2 if k % 2 else 1)
            for a in range(0, 360, 15):
                t = math.radians(a)
                d.line((cx + math.cos(t) * 0.16 * unit, cy + math.sin(t) * 0.16 * unit,
                        cx + math.cos(t) * 0.36 * unit, cy + math.sin(t) * 0.36 * unit),
                       fill=GOLD + (45,), width=1)
        # Constellations: a few joined stars.
        for _ in range(9):
            cx, cy = r.uniform(0, w), r.uniform(0, h * 0.75)
            pts = [(cx, cy)]
            for _ in range(r.randint(3, 6)):
                ang = r.uniform(0, math.tau)
                dist = r.uniform(0.03, 0.08) * unit
                pts.append((pts[-1][0] + math.cos(ang) * dist,
                            pts[-1][1] + math.sin(ang) * dist))
            d.line(pts, fill=GOLD + (110,), width=1)
            for px, py in pts:
                s = r.uniform(2.0, 3.6)
                d.ellipse((px - s, py - s, px + s, py + s), fill=GOLD + (220,))
        # Loose stars.
        for _ in range(int(w * h / 9000)):
            px, py = r.uniform(0, w), r.uniform(0, h * 0.8)
            s = r.choice((0.8, 1.0, 1.4))
            d.ellipse((px - s, py - s, px + s, py + s),
                      fill=CREAM + (r.randint(60, 180),))
        img = Image.alpha_composite(img.convert("RGBA"), lines).convert("RGB")
        return np.asarray(img)[:, :, ::-1].copy()

    def update(self, now):
        if self._last is None:
            self._last = now
            return
        ticks = min(int((now - self._last) / self.TICK_SECONDS), 40)
        if not ticks:
            return
        self._last += ticks * self.TICK_SECONDS
        for i, c in enumerate(self.clouds):
            c["x"] += c["speed"] * self.TICK_SECONDS * ticks
            if c["x"] < -0.4 or c["x"] > 1.4:
                self.clouds[i] = self._new_cloud()
        self._frame = None

    def render(self, w, h):
        if self._frame is not None and self._frame_size == (w, h):
            return self._frame
        if self._base_size != (w, h):
            self._base = self._make_base(w, h)
            self._base_size = (w, h)
        frame = self._base.copy()
        for c in self.clouds:
            colour, mask, ox, oy = self._sprite(c, h)
            px = int(round(c["x"] * w)) + ox
            py = int(round(c["y"] * h)) + oy
            mh, mw = mask.shape
            x0, y0 = max(0, px), max(0, py)
            x1, y1 = min(w, px + mw), min(h, py + mh)
            if x1 <= x0 or y1 <= y0:
                continue
            a = mask[y0 - py:y1 - py, x0 - px:x1 - px, None]
            col = colour[y0 - py:y1 - py, x0 - px:x1 - px]
            roi = frame[y0:y1, x0:x1].astype(np.float32)
            frame[y0:y1, x0:x1] = (roi + (col - roi) * a).astype(np.uint8)
        self._frame = frame
        self._frame_size = (w, h)
        self.version += 1
        return self._frame


# --- the archway ----------------------------------------------------------------
STAGE_HEIGHT_FRACTION = 0.94    # leaves sky above the arch for the ribbon


def dream_stage_rect(screen_w, screen_h, aspect):
    """The stage, a little shorter than the screen and standing on its floor."""
    max_h = screen_h * STAGE_HEIGHT_FRACTION
    w = screen_w
    h = int(round(w / aspect))
    if h > max_h:
        h = int(round(max_h))
        w = int(round(h * aspect))
    return (screen_w - w) // 2, screen_h - h, w, h


class Archway:
    """The poster's pink archway, framing the picture as a portal."""

    def __init__(self):
        self._key = None
        self.mask = None
        self.layer = None

    def geometry(self, w, h):
        band = max(6, int(round(min(w, h) * 0.04)))
        rx = w / 2.0
        ry = min(w / 2.0, h * 0.34)
        return band, rx, ry

    def prepare(self, screen_w, screen_h, rect):
        key = (screen_w, screen_h, rect)
        if key == self._key:
            return
        sx, sy, w, h = rect
        band, rx, ry = self.geometry(w, h)
        ss = 2
        # Inner opening: the picture shows through here.
        inner = Image.new("L", (w * ss, h * ss), 0)
        d = ImageDraw.Draw(inner)
        d.ellipse((band * ss, band * ss, (w - band) * ss, (2 * ry - band) * ss), fill=255)
        d.rectangle((band * ss, ry * ss, (w - band) * ss, h * ss), fill=255)
        inner = inner.resize((w, h), Image.LANCZOS)
        self.mask = (np.asarray(inner) > 127)[:, :, None]
        # The opening as row spans: below the spring line every row is the same
        # span, so the bulk of the picture is one slice copy; only the curved
        # rows above it need copying one at a time.
        m = self.mask[:, :, 0]
        self.full_from = int(math.ceil(ry))
        self.full_span = (band, w - band)
        self.rows = []
        for y in range(min(self.full_from, h)):
            xs = np.flatnonzero(m[y])
            if xs.size:
                self.rows.append((y, int(xs[0]), int(xs[-1]) + 1))

        # The band itself: salmon stone, fluted, with a cream inner edge.
        art = Image.new("RGBA", (screen_w * 1, screen_h * 1), (0, 0, 0, 0))
        layer = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        outer_box = (0, 0, w * ss, 2 * ry * ss)
        # Shadow cast outward onto the sky.
        shadow = Image.new("L", (w * ss, h * ss), 0)
        sd = ImageDraw.Draw(shadow)
        sd.ellipse(outer_box, fill=140)
        sd.rectangle((0, ry * ss, w * ss, h * ss), fill=140)
        d.ellipse(outer_box, fill=SALMON + (255,))
        d.rectangle((0, ry * ss, w * ss, h * ss), fill=SALMON + (255,))
        # Fluting on the pillars.
        for k in range(1, 4):
            x = int(band * ss * k / 4)
            d.line((x, ry * ss, x, h * ss), fill=(214, 128, 112, 255), width=ss)
            d.line(((w - band) * ss + x, ry * ss, (w - band) * ss + x, h * ss),
                   fill=(214, 128, 112, 255), width=ss)
        # Capitals where the arch springs.
        cap = int(band * 0.55)
        for x0 in (0, (w - band) * ss):
            d.rectangle((x0 - cap * ss // 3, (ry - cap) * ss, x0 + band * ss + cap * ss // 3,
                         ry * ss), fill=BLUSH + (255,))
            d.line((x0 - cap * ss // 3, ry * ss, x0 + band * ss + cap * ss // 3, ry * ss),
                   fill=(200, 118, 104, 255), width=ss * 2)
        # Keystone.
        kw = band * 1.3
        d.polygon(((w / 2 - kw * 0.6) * ss, 0, (w / 2 + kw * 0.6) * ss, 0,
                   (w / 2 + kw * 0.4) * ss, band * 1.2 * ss,
                   (w / 2 - kw * 0.4) * ss, band * 1.2 * ss), fill=BLUSH + (255,))
        # Cut the opening back out and line it with cream.
        hole = Image.new("L", (w * ss, h * ss), 0)
        hd = ImageDraw.Draw(hole)
        hd.ellipse((band * ss, band * ss, (w - band) * ss, (2 * ry - band) * ss), fill=255)
        hd.rectangle((band * ss, ry * ss, (w - band) * ss, h * ss), fill=255)
        la = np.asarray(layer).copy()
        hole_a = np.asarray(hole)
        la[hole_a > 0, 3] = 0
        layer = Image.fromarray(la)
        rim = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
        rd = ImageDraw.Draw(rim)
        rd.arc((band * ss, band * ss, (w - band) * ss, (2 * ry - band) * ss),
               180, 360, fill=CREAM + (255,), width=3 * ss)
        rd.line((band * ss + ss, ry * ss, band * ss + ss, h * ss), fill=CREAM + (255,),
                width=3 * ss)
        rd.line(((w - band) * ss - ss, ry * ss, (w - band) * ss - ss, h * ss),
                fill=CREAM + (255,), width=3 * ss)
        layer = Image.alpha_composite(layer, rim)
        layer = layer.resize((w, h), Image.LANCZOS)
        shadow = shadow.resize((w, h), Image.LANCZOS).filter(
            ImageFilter.GaussianBlur(band * 0.5))
        shadow_rgba = Image.new("RGBA", (w, h), INK + (0,))
        shadow_rgba.putalpha(shadow)
        # Shadow only outside the band, and only where it is not the opening.
        sh = np.asarray(shadow_rgba).copy()
        sh[np.asarray(inner) > 0, 3] = 0
        art.alpha_composite(Image.fromarray(sh), (sx, sy))
        art.alpha_composite(layer, (sx, sy))
        self.layer = SparseLayer(np.asarray(art))
        self._key = key

    def paste(self, canvas, stage, rect):
        sx, sy, w, h = rect
        a, b = self.full_span
        y = self.full_from
        canvas[sy + y:sy + h, sx + a:sx + b] = stage[y:h, a:b]
        for y, x0, x1 in self.rows:
            canvas[sy + y, sx + x0:sx + x1] = stage[y, x0:x1]

    def draw(self, canvas):
        self.layer.draw(canvas)


# --- the ribbon -----------------------------------------------------------------
class Backdrop:
    """Sky + archway + small print + dial, composed once per change.

    None of these move faster than ten times a second (the dial needle only
    while tuning), and the picture is pasted over the opening afterwards, so
    the whole surround is rebuilt only when one of its parts changes and is
    otherwise a single copy per frame.
    """

    def __init__(self):
        self._key = None
        self._frame = None

    def compose(self, sky, arch, extras):
        """`extras`: [(rgba ndarray, x, y)] blitted on top, by identity."""
        key = (id(sky), sky.shape, id(arch.layer),
               tuple((id(img), x, y) for img, x, y in extras))
        if key != self._key:
            frame = sky.copy()
            arch.draw(frame)
            for img, x, y in extras:
                blit_rgba(frame, img, x, y)
            self._frame = frame
            self._key = key
        return self._frame


def _star(d, cx, cy, r, fill):
    pts = []
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append((cx + math.cos(ang) * rr, cy + math.sin(ang) * rr))
    d.polygon(pts, fill=fill)


def ribbon_image(width, text="TUNE IN, THE DREAM IS REAL"):
    """The poster's banner: coral ribbon, notched tails, gold stars, cream type."""
    ss = 3
    W = int(width) * ss
    H = int(width * 0.105) * ss
    img = Image.new("RGBA", (W, int(H * 1.45)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    top = int(H * 0.2)
    tail = int(H * 0.5)
    dark = (150, 50, 58, 255)
    # Tails, folded behind.
    for side in (-1, 1):
        x_out = 0 if side < 0 else W
        x_in = tail * 1.6 if side < 0 else W - tail * 1.6
        y0, y1 = top + H * 0.22, top + H * 1.12
        notch = x_out + (tail * 0.5 if side < 0 else -tail * 0.5)
        d.polygon(((x_out, y0), (x_in, y0), (x_in, y1), (x_out, y1),
                   (notch, (y0 + y1) / 2)), fill=dark)
    body = (tail, top, W - tail, top + H * 0.9)
    d.polygon(((body[0], body[1]), (body[2], body[1]),
               (body[2] + tail * 0.35, (body[1] + body[3]) / 2), (body[2], body[3]),
               (body[0], body[3]), (body[0] - tail * 0.35, (body[1] + body[3]) / 2)),
              fill=CORAL + (255,))
    inset = H * 0.08
    d.line(((body[0] + inset, body[1] + inset), (body[2] - inset, body[1] + inset)),
           fill=CREAM + (150,), width=ss)
    d.line(((body[0] + inset, body[3] - inset), (body[2] - inset, body[3] - inset)),
           fill=CREAM + (150,), width=ss)
    f = font(int(H * 0.46))
    box = d.textbbox((0, 0), text, font=f)
    tw, th = box[2] - box[0], box[3] - box[1]
    cy = (body[1] + body[3]) / 2
    d.text((W / 2 - tw / 2 - box[0], cy - th / 2 - box[1]), text, font=f,
           fill=CREAM + (255,))
    for sx in (body[0] + H * 0.45, body[2] - H * 0.45):
        _star(d, sx, cy, H * 0.2, GOLD + (255,))
    img = img.resize((W // ss, img.height // ss), Image.LANCZOS)
    # A soft drop shadow, as if pasted on.
    a = img.split()[3].filter(ImageFilter.GaussianBlur(3))
    shadow = Image.new("RGBA", img.size, INK + (0,))
    shadow.putalpha(a.point(lambda v: int(v * 0.5)))
    out = Image.new("RGBA", (img.width + 8, img.height + 8), (0, 0, 0, 0))
    out.alpha_composite(shadow, (4, 7))
    out.alpha_composite(img, (2, 2))
    return out


# --- the radio dial -------------------------------------------------------------
class RadioDial:
    """A vintage tuning panel: frequency, station, scale and needle, signal."""

    def __init__(self):
        self._key = None
        self._img = None

    def size_for(self, screen_w, screen_h, margin):
        w = int(min(max(margin * 0.86, screen_h * 0.3), screen_h * 0.42))
        return w, int(w * 0.62)

    def render(self, w, h, freq, name, on_air, signal, now):
        blink = int(now * 2) % 2 if on_air else 0
        bars = int(round(signal * 5))
        key = (w, h, round(freq, 1), name, on_air, blink, bars)
        if key == self._key:
            return self._img
        ss = 2
        img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        r = 18 * ss
        d.rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=r,
                            fill=(33, 30, 36, 225), outline=GOLD + (200,), width=2 * ss)
        pad = int(w * 0.06) * ss
        # Station call sign.
        small = font(int(h * 0.085) * ss)
        d.text((pad, pad * 0.8), "DREAMWAVE FM", font=small, fill=GOLD + (255,))
        if on_air:
            tag = "● ON AIR"
            colour = CORAL + (255 if blink else 140,)
            tb = d.textbbox((0, 0), tag, font=small)
            d.text((w * ss - pad - (tb[2] - tb[0]), pad * 0.8), tag, font=small,
                   fill=colour)
        # Frequency, in an LCD face.
        big = font(int(h * 0.34) * ss, DIAL_FONT)
        ftxt = f"{freq:5.1f}"
        d.text((pad, h * 0.16 * ss), ftxt, font=big, fill=CREAM + (255,))
        fb = d.textbbox((pad, h * 0.16 * ss), ftxt, font=big)
        d.text((fb[2] + 6 * ss, fb[3] - h * 0.13 * ss), "MHz",
               font=font(int(h * 0.09) * ss, DIAL_FONT), fill=CREAM + (170,))
        # Signal bars.
        bx = w * ss - pad
        for k in range(5):
            bh = (k + 1) * h * 0.035 * ss
            x1 = bx - (4 - k) * w * 0.032 * ss
            x0 = x1 - w * 0.02 * ss
            on = k < bars
            d.rectangle((x0, fb[3] - bh, x1, fb[3]),
                        fill=(GOLD + (255,)) if on else (CREAM + (50,)))
        # Station name.
        nfont = font(int(h * 0.105) * ss)
        label = f"“{name}”" if on_air else name
        limit = w * ss - pad * 2
        while len(label) > 4 and d.textlength(label, font=nfont) > limit:
            label = label[:-2].rstrip() + "…"
        d.text((pad, h * 0.56 * ss), label, font=nfont,
               fill=(CORAL if on_air else BLUSH) + (255,))
        # Scale with ticks and the needle.
        y = h * 0.82 * ss
        x0, x1 = pad, w * ss - pad
        d.line((x0, y, x1, y), fill=CREAM + (150,), width=ss)
        for k in range(int(DIAL_MIN) + 1, int(DIAL_MAX) + 1):
            x = x0 + (k - DIAL_MIN) / (DIAL_MAX - DIAL_MIN) * (x1 - x0)
            tall = k % 4 == 0
            d.line((x, y - (8 if tall else 4) * ss, x, y), fill=CREAM + (150,), width=ss)
            if tall:
                lab = str(k)
                lf = font(int(h * 0.085) * ss, DIAL_FONT)
                lb = d.textbbox((0, 0), lab, font=lf)
                d.text((x - (lb[2] - lb[0]) / 2, y + 3 * ss), lab, font=lf,
                       fill=CREAM + (130,))
        for st in STATIONS:
            x = x0 + (st.freq - DIAL_MIN) / (DIAL_MAX - DIAL_MIN) * (x1 - x0)
            d.ellipse((x - 2 * ss, y - 2 * ss, x + 2 * ss, y + 2 * ss), fill=GOLD + (220,))
        nx = x0 + (min(max(freq, DIAL_MIN), DIAL_MAX) - DIAL_MIN) / (DIAL_MAX - DIAL_MIN) * (x1 - x0)
        d.line((nx, y - h * 0.16 * ss, nx, y + h * 0.05 * ss), fill=CORAL + (255,),
               width=3 * ss)
        self._img = np.asarray(img.resize((w, h), Image.LANCZOS))
        self._key = key
        return self._img


# --- the coin -------------------------------------------------------------------
def emblem_faces(size):
    """Both faces of the coin: a gold eye on a radio mast, and its coral back.

    Returns {"front": (bgr, alpha), "back": (bgr, alpha)} at `size` pixels.
    """
    ss = 4
    S = size * ss
    faces = {}
    for name, disc, ink, rim in (("front", GOLD, INK, (245, 214, 140)),
                                 ("back", CORAL, CREAM, BLUSH)):
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((0, 0, S - 1, S - 1), fill=disc + (255,))
        d.ellipse((S * 0.05, S * 0.05, S * 0.95, S * 0.95), outline=rim + (255,),
                  width=max(1, int(S * 0.025)))
        w = max(1, int(S * 0.035))
        cx, cy = S / 2, S * 0.43
        if name == "front":
            # The mast.
            d.line((cx, cy + S * 0.1, cx - S * 0.16, S * 0.86), fill=ink + (255,), width=w)
            d.line((cx, cy + S * 0.1, cx + S * 0.16, S * 0.86), fill=ink + (255,), width=w)
            for k in range(1, 4):
                t = k / 4
                yy = cy + S * 0.1 + t * (S * 0.86 - cy - S * 0.1)
                d.line((cx - S * 0.16 * t, yy, cx + S * 0.16 * t, yy), fill=ink + (255,),
                       width=max(1, w // 2))
            # The eye.
            ew, eh = S * 0.2, S * 0.11
            d.chord((cx - ew, cy - eh * 1.4, cx + ew, cy + eh * 1.4), 200, 340,
                    fill=CREAM + (255,))
            d.ellipse((cx - ew, cy - eh, cx + ew, cy + eh), fill=CREAM + (255,),
                      outline=ink + (255,), width=w)
            d.ellipse((cx - eh * 0.75, cy - eh * 0.75, cx + eh * 0.75, cy + eh * 0.75),
                      fill=ink + (255,))
            d.ellipse((cx - eh * 0.2 + eh * 0.25, cy - eh * 0.45, cx + eh * 0.2 + eh * 0.25,
                       cy - eh * 0.05), fill=CREAM + (255,))
            # Waves.
            for k in range(1, 3):
                rr = S * (0.25 + 0.08 * k)
                for a0, a1 in ((200, 250), (290, 340)):
                    d.arc((cx - rr, cy - rr, cx + rr, cy + rr), a0, a1, fill=ink + (255,),
                          width=w)
        else:
            _star(d, cx, S * 0.5, S * 0.28, ink + (255,))
            for k in range(3):
                rr = S * (0.33 + 0.04 * k)
                d.arc((cx - rr, S * 0.5 - rr, cx + rr, S * 0.5 + rr), 200, 340,
                      fill=ink + (200,), width=max(1, w // 2))
        img = img.resize((size, size), Image.LANCZOS)
        a = np.asarray(img)
        faces[name] = (np.ascontiguousarray(a[:, :, 2::-1]),
                       a[:, :, 3].astype(np.float32) / 255.0)
    return faces


class Glow:
    """The soft gold halo behind the coin, breathing, brighter on air."""

    def __init__(self):
        self._key = None
        self._img = None

    def draw(self, canvas, cx, cy, radius, now, boost=0.0):
        size = int(radius * 4) | 1
        if self._key != size:
            g = np.zeros((size, size), np.float32)
            cv2.circle(g, (size // 2, size // 2), int(radius * 0.9), 1.0, -1, cv2.LINE_AA)
            g = cv2.GaussianBlur(g, (0, 0), radius * 0.55)
            g /= max(1e-6, g.max())
            rgba = np.zeros((size, size, 4), np.uint8)
            rgba[:, :, :3] = GOLD
            rgba[:, :, 3] = (g * 255).astype(np.uint8)
            self._img = rgba
            self._key = size
        pulse = 0.45 + 0.2 * math.sin(now * 2.2) + 0.35 * boost
        blit_rgba(canvas, self._img, int(cx - size // 2), int(cy - size // 2),
                  opacity=min(1.0, pulse))


# --- static ---------------------------------------------------------------------
class Static:
    """Analogue snow and a rolling bar, for the moment between stations."""

    def __init__(self, seed=5):
        self.rng = np.random.default_rng(seed)

    def apply(self, stage, level, now):
        if level <= 0.01:
            return stage
        h, w = stage.shape[:2]
        sw, sh = max(32, w // 3), max(32, h // 3)
        snow = self.rng.integers(0, 256, (sh, sw), dtype=np.uint8)
        snow = cv2.resize(snow, (w, h), interpolation=cv2.INTER_NEAREST)
        snow = cv2.cvtColor(snow, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(stage, 1.0 - level, snow, level, 0.0)
        # The rolling bar, and a few lines pushed sideways.
        bar_y = int(((now * 0.9) % 1.0) * h)
        bh = max(4, h // 14)
        y0, y1 = max(0, bar_y - bh), min(h, bar_y + bh)
        out[y0:y1] = cv2.addWeighted(out[y0:y1], 0.75, np.full_like(out[y0:y1], 235), 0.25 * level * 2, 0)
        for _ in range(int(6 * level)):
            y = int(self.rng.integers(0, h - 6))
            shift = int(self.rng.integers(-w // 20, w // 20))
            out[y:y + 5] = np.roll(out[y:y + 5], shift, axis=1)
        return out


# --- small print ------------------------------------------------------------------
def footer_image(height, lines=("nothing here is recorded or stored",
                                "london decompression · dreamwave 2026")):
    f = font(max(10, int(height * 0.0145)))
    pad = 4
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    boxes = [measure.textbbox((0, 0), t, font=f) for t in lines]
    lh = max(b[3] - b[1] for b in boxes)
    w = max(b[2] - b[0] for b in boxes) + pad * 2
    h = (lh + int(lh * 0.5)) * len(lines) + pad * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = pad
    for t, b in zip(lines, boxes):
        d.text((w - pad - (b[2] - b[0]) - b[0], y - b[1]), t, font=f, fill=CREAM + (150,))
        y += lh + int(lh * 0.5)
    return np.asarray(img)
