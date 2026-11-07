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
# No "poster", "card", "portrait frame" or "archway": the picture is already
# shown through an arch, and those words make the model draw a second frame or
# arch inside it.
STYLE = ("surreal dreamlike vintage paper collage, halftone print texture, rich "
         "colours, full bleed")


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
            "full body view of a dreamer whose body is made of soft white clouds, "
            "vivid blue sky behind",
            "a kingdom of floating cloud islands in a vivid blue sky, golden light"),
    Station(90.7, "Radio Saints",
            "full body view of a radiant saint with a halo of radio waves and "
            "golden constellations on midnight blue, ornate gold embroidery",
            "a midnight cathedral of radio towers under golden constellations"),
    Station(93.3, "Moon Ballroom",
            "full body view of an elegant dancer in a silver sequined gown, giant "
            "moon and mirror ball, deep violet night",
            "a ballroom on the surface of the moon, giant mirror ball, violet night sky"),
    Station(95.9, "Deep Sea Static",
            "full body view of a glowing deep sea jellyfish person with "
            "bioluminescent tendrils, teal and turquoise underwater",
            "a glowing coral reef city deep under a turquoise sea"),
    Station(98.4, "Playa Mirage",
            "full body view of a desert wanderer in a fantastical fur coat and "
            "goggles, dust storm at golden hour, glowing art car",
            "an endless desert playa at golden hour with glowing art cars and a giant effigy"),
    Station(101.1, "The Hypnotist",
            "full body view of a mysterious hypnotist with an all-seeing eye, black "
            "and cream op-art spiral, red velvet curtains",
            "an endless hypnotic landscape of black and cream op-art spirals under a red sky"),
    Station(103.7, "Tin Radio Heart",
            "full body view of a vintage tin toy robot with a glowing radio dial for "
            "a heart, mint green and chrome",
            "a retro-futurist city of chrome radio towers and mint green skies"),
    Station(106.5, "Garden of Lost Keys",
            "full body view of a forest spirit made of flowers, antique keys and "
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
    return f"full body view of {subject}, {STYLE}"


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
CLOUD_DIR = ASSETS_DIR / "clouds"


def _load_cloud_sprites():
    """Photographic cloud cut-outs (BGRA), as in the poster's collage.

    Generated once with the mirror's own model as white cumulus on black and
    keyed out, so they carry their soft, wispy edges.
    """
    sprites = []
    for path in sorted(CLOUD_DIR.glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            # Soften whatever edge the cloud was cropped against.
            h = img.shape[0]
            fade = np.ones(h, np.float32)
            n = max(2, h // 8)
            fade[-n:] = np.linspace(1.0, 0.0, n) ** 1.5
            img[:, :, 3] = (img[:, :, 3] * fade[:, None]).astype(np.uint8)
            sprites.append(img)
    return sprites


def _procedural_cloud(rng, size=360):
    """Fallback sprite if the cut-outs are missing: overlapping soft puffs."""
    w, h = size, int(size * 0.6)
    alpha = np.zeros((h, w), np.float32)
    for _ in range(14):
        r = rng.uniform(0.12, 0.26) * w
        cx = rng.uniform(0.25, 0.75) * w
        cy = h * 0.65 - rng.uniform(0, 0.35) * h * (1 - abs(cx / w - 0.5))
        cv2.circle(alpha, (int(cx), int(cy)), int(r), 1.0, -1, cv2.LINE_AA)
    alpha[int(h * 0.82):] = 0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 3)
    shade = np.linspace(255, 205, h, dtype=np.float32)[:, None]
    bgr_ = np.dstack([shade * np.ones((1, w), np.float32)] * 3)
    return np.dstack([bgr_, alpha * 255]).astype(np.uint8)


class DuskSky:
    """The sky behind the arch.

    A dusk gradient; warm light and faint rays spilling from behind the arch,
    as if the portal were lit; gold constellations and astrolabe rings; a
    halftone horizon and paper grain for the printed-collage feel; and
    photographic clouds drifting across in two layers, far ones smaller,
    dimmer and slower, all lit pink from underneath by the sunset.

    The base is drawn once per screen size; each cloud is prepared once as a
    lit sprite with a soft paper shadow and then only moved, ten times a
    second, blended into just the rectangle it covers.
    """

    TICK_SECONDS = 0.1
    COUNT = 11

    def __init__(self, seed=11):
        self.rng = random.Random(seed)
        self.sprites = _load_cloud_sprites() or [_procedural_cloud(self.rng)
                                                 for _ in range(4)]
        self.clouds = [self._new_cloud(layer=i % 2, initial=True)
                       for i in range(self.COUNT)]
        self.portal = None
        self._base = None
        self._base_key = None
        self._frame = None
        self._frame_size = None
        self._last = None

    def set_portal(self, rect):
        """Where the arch stands, so the light can come from behind it."""
        if rect != self.portal:
            self.portal = rect
            self._frame = None

    def _new_cloud(self, layer, initial=False):
        r = self.rng
        far = layer == 0
        speed = r.uniform(0.003, 0.006) if far else r.uniform(0.006, 0.012)
        return {
            "sprite": r.randrange(len(self.sprites)),
            "flip": r.random() < 0.5,
            "height": r.uniform(0.12, 0.19) if far else r.uniform(0.22, 0.36),
            "x": r.uniform(-0.1, 1.1) if initial else (-0.3 if speed > 0 else 1.3),
            "y": r.uniform(0.04, 0.5) if far else r.uniform(0.3, 0.92),
            "speed": speed * (1 if r.random() < 0.5 else -1),
            "dim": r.uniform(0.78, 0.88) if far else r.uniform(0.96, 1.06),
            "opacity": 0.85 if far else 1.0,
            "lit": None,
        }

    def _lit(self, c, h):
        """The cloud's sprite at its size, lit for dusk, with a paper shadow."""
        if c["lit"] is not None and c["lit"][0] == h:
            return c["lit"][1:]
        src = self.sprites[c["sprite"]]
        if c["flip"]:
            src = src[:, ::-1]
        th = max(8, int(c["height"] * h))
        tw = max(8, int(src.shape[1] * th / src.shape[0]))
        spr = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
        col, a = spr[:, :, :3], spr[:, :, 3] / 255.0 * c["opacity"]
        # Sunset from below: cream on top, pink and a little violet underneath.
        rows = np.linspace(0.0, 1.0, th, dtype=np.float32)[:, None, None]
        top = np.array((228, 242, 255), np.float32) / 255.0          # BGR, warm cream
        under = np.array((178, 160, 255), np.float32) / 255.0        # BGR, sunset pink
        light = top + (under - top) * np.clip((rows - 0.25) / 0.75, 0, 1)
        col = col * light * c["dim"]
        # A soft shadow below and to the right, as if pasted on. Everything is
        # stored premultiplied: frame = frame * (1 - alpha) + colour.
        pad = max(6, th // 7)
        H2, W2 = th + pad * 2, tw + pad * 2
        ca = np.zeros((H2, W2), np.float32)
        ca[pad:pad + th, pad:pad + tw] = a
        cc = np.zeros((H2, W2, 3), np.float32)
        cc[pad:pad + th, pad:pad + tw] = col * a[:, :, None]
        shadow = np.zeros((H2, W2), np.float32)
        dy, dx = pad // 4, pad // 6
        shadow[pad + dy:pad + dy + th, pad + dx:pad + dx + tw] = a
        shadow = cv2.GaussianBlur(shadow, (0, 0), pad * 0.22) * 0.22 * (1 - ca)
        alpha = ca + shadow
        colour = cc + np.array(bgr(INK), np.float32) * shadow[:, :, None]
        c["lit"] = (h, colour, alpha[:, :, None],
                    -W2 // 2, -H2 // 2)
        return c["lit"][1:]

    def _make_base(self, w, h):
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        stops = ((0.0, (20, 20, 40)), (0.45, (58, 36, 70)), (0.75, (122, 62, 86)),
                 (1.0, (198, 106, 98)))
        pos = np.array([p for p, _ in stops], np.float32)
        base = np.empty((h, w, 3), np.float32)
        for c in range(3):
            vals = np.array([col[2 - c] for _, col in stops], np.float32)
            base[:, :, c] = np.interp(ys, pos, vals)

        # Light from behind the arch: a warm glow and faint rays.
        if self.portal is not None:
            px, py, pw, ph = self.portal
            cx, cy = px + pw / 2.0, py + ph * 0.32
        else:
            cx, cy = w / 2.0, h * 0.35
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        dist = np.sqrt(dx * dx + dy * dy) / h
        glow = np.exp(-(dist / 0.5) ** 2) * 0.32
        warm = np.array(bgr((255, 150, 150)), np.float32)
        base += (warm - base) * glow[:, :, None]
        angle = np.arctan2(dy, dx)
        rays = (np.cos(angle * 18) > 0.55).astype(np.float32)
        rays = cv2.GaussianBlur(rays, (0, 0), 2.0) * np.exp(-(dist / 0.75) ** 2) * 0.07
        base += (np.array(bgr(CREAM), np.float32) - base) * rays[:, :, None]

        # Halftone horizon: coral dots growing towards the bottom edge.
        tone = np.zeros((h, w), np.float32)
        pitch = max(6, h // 120)
        top = int(h * 0.68)
        for y in range(top, h + pitch, pitch):
            k = (y - top) / max(1, h - top)
            rad = pitch * 0.5 * k ** 1.3
            if rad < 0.6:
                continue
            off = (pitch // 2) if ((y - top) // pitch) % 2 else 0
            for x in range(off, w + pitch, pitch):
                cv2.circle(tone, (x, y), int(round(rad)), 1.0, -1, cv2.LINE_AA)
        tone *= 0.28
        dots = np.array(bgr((246, 150, 128)), np.float32)
        base += (dots - base) * tone[:, :, None]

        img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)[:, :, ::-1], "RGB")
        lines = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(lines)
        r = random.Random(3)
        unit = h
        # Astrolabe rings up in the corners, clear of the dial and small print.
        for (ax, ay) in ((0.07 * w, 0.16 * h), (0.93 * w, 0.2 * h)):
            for k, rad in enumerate((0.12, 0.17, 0.23, 0.27)):
                rr = rad * unit
                d.ellipse((ax - rr, ay - rr, ax + rr, ay + rr),
                          outline=GOLD + (70 if k % 2 else 45,), width=1)
            for a in range(0, 360, 10):
                t = math.radians(a)
                d.line((ax + math.cos(t) * 0.23 * unit, ay + math.sin(t) * 0.23 * unit,
                        ax + math.cos(t) * 0.27 * unit, ay + math.sin(t) * 0.27 * unit),
                       fill=GOLD + (55,), width=1)
        for _ in range(8):
            sx_, sy_ = r.uniform(0, w), r.uniform(0, h * 0.6)
            pts = [(sx_, sy_)]
            for _ in range(r.randint(3, 5)):
                ang = r.uniform(0, math.tau)
                dd = r.uniform(0.03, 0.07) * unit
                pts.append((pts[-1][0] + math.cos(ang) * dd, pts[-1][1] + math.sin(ang) * dd))
            d.line(pts, fill=GOLD + (80,), width=1)
            for qx, qy in pts:
                rad = r.uniform(1.6, 3.0)
                d.ellipse((qx - rad, qy - rad, qx + rad, qy + rad), fill=GOLD + (210,))
        for _ in range(int(w * h / 7000)):
            qx, qy = r.uniform(0, w), r.uniform(0, h * 0.7)
            rad = r.choice((0.7, 0.9, 1.3))
            d.ellipse((qx - rad, qy - rad, qx + rad, qy + rad),
                      fill=CREAM + (r.randint(50, 170),))
        img = Image.alpha_composite(img.convert("RGBA"), lines).convert("RGB")
        out = np.asarray(img)[:, :, ::-1].astype(np.float32)
        # Paper grain.
        grain = np.random.default_rng(9).normal(0, 4.0, (h, w)).astype(np.float32)
        out += grain[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8)

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
            if c["x"] < -0.35 or c["x"] > 1.35:
                layer = 0 if c["height"] < 0.2 else 1
                self.clouds[i] = self._new_cloud(layer)
        self._frame = None

    def render(self, w, h):
        if self._frame is not None and self._frame_size == (w, h):
            return self._frame
        key = (w, h, self.portal)
        if self._base_key != key:
            self._base = self._make_base(w, h)
            self._base_key = key
        frame = self._base.copy()
        # Far layer first.
        for c in sorted(self.clouds, key=lambda c: c["height"]):
            colour, alpha, ox, oy = self._lit(c, h)
            px = int(round(c["x"] * w)) + ox
            py = int(round(c["y"] * h)) + oy
            mh, mw = alpha.shape[:2]
            x0, y0 = max(0, px), max(0, py)
            x1, y1 = min(w, px + mw), min(h, py + mh)
            if x1 <= x0 or y1 <= y0:
                continue
            a = alpha[y0 - py:y1 - py, x0 - px:x1 - px]
            col = colour[y0 - py:y1 - py, x0 - px:x1 - px]
            roi = frame[y0:y1, x0:x1].astype(np.float32)
            frame[y0:y1, x0:x1] = np.clip(roi * (1.0 - a) + col, 0, 255).astype(np.uint8)
        self._frame = frame
        self._frame_size = (w, h)
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


def _bump(t, centre, width):
    return np.exp(-((t - centre) / width) ** 2)


class Archway:
    """The poster's pink archway, built as carved stone around the picture.

    The shape is turned into a height map — a moulded profile around the arch
    (raised lips at both edges, a gentle swell between), fluted pilasters,
    capitals where the arch springs, plinths at the foot and a keystone — and
    lit from the upper left, so it reads as stone rather than a flat band. It
    casts a soft shadow out onto the sky and another in onto the picture, which
    is what sets the picture back *behind* it, through the opening.
    """

    LIGHT = np.array((-0.45, -0.65, 0.62), np.float32)

    def __init__(self):
        self._key = None
        self.mask = None
        self.layer = None
        self.keep = None

    def geometry(self, w, h):
        band = max(8, int(round(min(w, h) * 0.05)))
        rx = w / 2.0
        ry = min(w / 2.0, h * 0.36)
        return band, rx, ry

    def prepare(self, screen_w, screen_h, rect):
        key = (screen_w, screen_h, rect)
        if key == self._key:
            return
        sx, sy, w, h = rect
        band, rx, ry = self.geometry(w, h)
        ss = 2
        P = int(band * 0.6)                      # room for what juts out
        W2, H2 = (w + 2 * P) * ss, (h + P) * ss
        ox, oy = P * ss, P * ss                  # stage origin in the layer
        B, R = band * ss, ry * ss

        def shape(inset):
            m = np.zeros((H2, W2), np.uint8)
            cv2.ellipse(m, (int(ox + w * ss / 2), int(oy + R)),
                        (int(w * ss / 2 - inset), int(R - inset)), 0, 180, 360, 255, -1,
                        cv2.LINE_AA)
            cv2.rectangle(m, (int(ox + inset), int(oy + R)),
                          (int(ox + w * ss - inset), H2), 255, -1)
            return m

        outer, inner = shape(0), shape(B)
        cap_h, cap_out = int(B * 0.75), int(B * 0.28)
        base_h, base_out = int(B * 1.1), int(B * 0.22)
        blocks = np.zeros((H2, W2), np.uint8)
        for x0 in (ox, ox + w * ss - B):
            cv2.rectangle(blocks, (int(x0 - cap_out), int(oy + R - cap_h)),
                          (int(x0 + B + cap_out), int(oy + R)), 255, -1)
            cv2.rectangle(blocks, (int(x0 - base_out), H2 - base_h),
                          (int(x0 + B + base_out), H2), 255, -1)
        # Nothing may jut into the opening: the picture shows there.
        blocks[inner > 127] = 0
        solid = np.maximum(outer, blocks)
        stone = (solid > 127) & (inner <= 127)

        # Height map.
        d_in = cv2.distanceTransform((inner <= 127).astype(np.uint8), cv2.DIST_L2, 5)
        d_out = cv2.distanceTransform((outer > 127).astype(np.uint8), cv2.DIST_L2, 5)
        t = np.clip(d_in / np.maximum(d_in + d_out, 1e-3), 0, 1)
        profile = (0.55 * _bump(t, 0.12, 0.08) + 0.55 * _bump(t, 0.88, 0.08)
                   + 0.3 * np.sin(np.pi * t))
        height = profile * B * 0.3
        yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
        pillar = (yy > oy + R + cap_h * 0.2) & (yy < H2 - base_h) & (outer > 127)
        # Fluting: three shallow grooves down each pilaster.
        for x0 in (ox, ox + w * ss - B):
            local = (xx - x0) / B
            inside = pillar & (local > 0.18) & (local < 0.82)
            groove = 0.5 + 0.5 * np.cos((local - 0.18) / 0.64 * 3 * 2 * np.pi)
            height[inside] -= (groove[inside] * B * 0.09)
        blk = cv2.GaussianBlur((blocks > 127).astype(np.float32), (0, 0), B * 0.06)
        height += blk * B * 0.38
        # Mouldings across the capitals and plinths.
        for y_line in (oy + R - cap_h * 0.55, H2 - base_h * 0.6):
            band_rows = np.exp(-((yy - y_line) / (B * 0.07)) ** 2)
            height += band_rows * blk * B * 0.12
        # Keystone.
        key_w = B * 1.25
        kp = np.array([(ox + w * ss / 2 - key_w * 0.6, oy - B * 0.15),
                       (ox + w * ss / 2 + key_w * 0.6, oy - B * 0.15),
                       (ox + w * ss / 2 + key_w * 0.42, oy + B * 1.05),
                       (ox + w * ss / 2 - key_w * 0.42, oy + B * 1.05)], np.int32)
        keyst = np.zeros((H2, W2), np.uint8)
        cv2.fillConvexPoly(keyst, kp, 255, cv2.LINE_AA)
        keyst[inner > 127] = 0
        stone |= keyst > 127
        height += cv2.GaussianBlur((keyst > 127).astype(np.float32), (0, 0), B * 0.05) * B * 0.32
        height = cv2.GaussianBlur(height, (0, 0), ss * 0.9)

        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        n = np.dstack([-gx, -gy, np.ones_like(gx)])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        L = self.LIGHT / np.linalg.norm(self.LIGHT)
        diffuse = np.clip((n * L).sum(axis=2), 0, 1)
        half = L + np.array((0, 0, 1), np.float32)
        half /= np.linalg.norm(half)
        spec = np.clip((n * half).sum(axis=2), 0, 1) ** 28
        edge = np.minimum(d_in, d_out)
        ao = 1.0 - 0.28 * np.exp(-edge / (B * 0.06))
        rng = np.random.default_rng(4)
        grain = cv2.GaussianBlur(rng.normal(0, 1, (H2, W2)).astype(np.float32), (0, 0), 1.2)
        mottle = cv2.GaussianBlur(rng.normal(0, 1, (H2 // 16 + 1, W2 // 16 + 1))
                                  .astype(np.float32), (0, 0), 1.5)
        mottle = cv2.resize(mottle, (W2, H2), interpolation=cv2.INTER_CUBIC)
        albedo = np.array(bgr(SALMON), np.float32) * (1.0 + 0.05 * grain + 0.05 * mottle)[:, :, None]
        shade = (0.5 + 0.68 * diffuse) * ao
        colour = albedo * shade[:, :, None] + spec[:, :, None] * 70.0
        colour = np.clip(colour, 0, 255)
        alpha = stone.astype(np.float32)

        # Shadow out onto the sky, down and to the right.
        so = int(B * 0.25)
        shadow = np.zeros((H2, W2), np.float32)
        shadow[so:, so:] = alpha[:-so, :-so]
        shadow = cv2.GaussianBlur(shadow, (0, 0), B * 0.4) * 0.4
        shadow[inner > 127] = 0
        out_alpha = alpha + shadow * (1 - alpha)
        out_col = (colour * alpha[:, :, None]
                   + np.array(bgr(INK), np.float32) * (shadow * (1 - alpha))[:, :, None])
        out_col = out_col / np.maximum(out_alpha, 1e-4)[:, :, None]

        # Shadow in onto the picture: the arch is in front of it.
        cast = np.zeros((H2, W2), np.float32)
        si = int(B * 0.35)
        cast[si:, si // 2:] = alpha[:-si, :-(si // 2) or None]
        cast = cv2.GaussianBlur(np.maximum(cast, alpha), (0, 0), B * 0.45)
        rim = np.exp(-d_in / (B * 0.25)) * 0.3
        inner_shadow = np.clip(cast * 0.6 + rim, 0, 0.62) * (inner > 127)

        # Down to screen resolution, into screen space.
        def down(a):
            return cv2.resize(a, (W2 // ss, H2 // ss), interpolation=cv2.INTER_AREA)
        lw, lh = W2 // ss, H2 // ss
        lx, ly = sx - P, sy - P
        art = np.zeros((screen_h, screen_w, 4), np.uint8)
        rgba = np.dstack([down(out_col)[:, :, ::-1], down(out_alpha) * 255])
        x0, y0 = max(0, lx), max(0, ly)
        x1, y1 = min(screen_w, lx + lw), min(screen_h, ly + lh)
        art[y0:y1, x0:x1] = np.clip(rgba[y0 - ly:y1 - ly, x0 - lx:x1 - lx], 0, 255)
        self.layer = SparseLayer(art)
        # The picture's own shadow is a darkening, applied as one multiply.
        keep = 1.0 - down(inner_shadow)[P:P + h, P:P + w]
        self.keep = np.repeat((keep * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        # The opening, at stage resolution, as row spans for the paste.
        opening = cv2.resize(inner, (lw, lh), interpolation=cv2.INTER_AREA)[P:P + h, P:P + w]
        self.mask = (opening > 127)[:, :, None]
        m = self.mask[:, :, 0]
        self.full_from = int(math.ceil(ry))
        self.full_span = (band, w - band)
        self.rows = []
        for y in range(min(self.full_from, h)):
            xs = np.flatnonzero(m[y])
            if xs.size:
                self.rows.append((y, int(xs[0]), int(xs[-1]) + 1))
        self._key = key

    def paste(self, canvas, stage, rect):
        sx, sy, w, h = rect
        a, b = self.full_span
        y = self.full_from
        canvas[sy + y:sy + h, sx + a:sx + b] = stage[y:h, a:b]
        for y, x0, x1 in self.rows:
            canvas[sy + y, sx + x0:sx + x1] = stage[y, x0:x1]

    def draw(self, canvas):
        """The stone and its shadow on the sky (baked into the backdrop)."""
        self.layer.draw(canvas)

    def shade(self, stage):
        """The picture with the arch's shadow cast onto it (a new array)."""
        if self.keep is None or self.keep.shape != stage.shape:
            return stage
        return cv2.multiply(stage, self.keep, scale=1.0 / 255.0)


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
