"""Dream World: the first step from a flat mirror towards a world you walk into.

The mirror redraws you where you stand. Dream World instead puts you *inside*
a generated place:

* **A world around you.** Each station (and each spoken dream) has a place as
  well as a look. It is generated once, given depth by Depth Anything, and
  drawn as a 2.5D scene with head-coupled parallax: move left and the near
  parts of the world slide past the far ones; step towards the mirror and the
  camera dollies in. It reads as looking through the archway into a space, not
  at a picture.
* **You, cut out cleanly.** With a Kinect the cut-out is metric: a per-pixel
  model of the empty room's depth (the floor, any furniture) is learned as the
  night goes on, and anything nearer than it is a person. Without a Kinect the
  cut-out comes from estimated depth. Your restyled self is composited into
  the world with a faint gold rim, as if you had stepped through.
* **Spoken things become objects in it.** Say "a giant jellyfish" and one is
  generated, cut out, and set floating in the world at a depth of its own,
  passing in front of you or behind you according to where you are standing.

All generation runs on the inference thread between batches (the GPU is shared,
and two threads driving MPS at once is asking for trouble), so the mirror never
stalls; a new world or object turns up a fraction of a second later.
"""

import math
import random
import threading
import time

import cv2
import numpy as np

import dreamwave

WORK_SCALE = 0.5            # parallax is computed at half the stage size
MAX_OBJECTS = 5
OBJECT_LIFETIME = 120.0
BACKDROP_FADE = 1.6
PARALLAX = 0.075            # fraction of the width the nearest layer can slide
DOLLY = 0.16                # how far stepping in pushes the camera forward


# --- who is a person ------------------------------------------------------------
class DepthBackground:
    """Learns the empty room's depth, so only people are cut out.

    Works on the uint8 depth maps the Kinect source produces (brighter is
    nearer, 0 is no reading). Anything clearly nearer than the learned room is
    a person — which takes care of the floor that creeps into the working
    volume, and of any furniture.

    The Kinect's no-reading pixels are the trap. Each person casts an IR shadow
    beside them that reads as "nothing there", i.e. far; learn that, and when
    they step away the floor that was in their shadow looks nearer than the
    background and is cut out as if it were a person. So:

    * while the room is **empty**, the background simply becomes the room as it
      is now (persistent no-reading regions included — that is real distance);
    * while **people are present**, only real readings can push the background
      farther (a revealed patch of floor); no-reading pixels never do;
    * and as a backstop, any surface unchanged for half a minute is absorbed.
    """

    MARGIN = 10.0               # map units, ~100mm at the default volume
    HOLE_FRAMES = 20            # no reading for this long counts as far, if empty
    EMPTY_FRAMES = 8            # consecutive near-empty frames to call the room empty
    EMPTY_FRACTION = 0.002
    SETTLE_FRAMES = 600         # ~30s of stillness and a surface is background
    STILL = 3.0

    def __init__(self):
        self.far = None
        self.last = None
        self.missing = None
        self.still = None
        self.empty_run = 0

    def masks(self, maps):
        out = []
        for m in maps:
            near = m[:, :, 0] if m.ndim == 3 else m
            # Fill speckle holes from their neighbours first.
            filled = cv2.morphologyEx(near, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            valid = filled > 0
            reading = 255.0 - filled.astype(np.float32)
            if self.far is None or self.far.shape != reading.shape:
                self.far = np.where(valid, reading, 255.0).astype(np.float32)
                self.last = self.far.copy()
                self.missing = np.zeros(reading.shape, np.int32)
                self.still = np.zeros(reading.shape, np.int32)
            self.missing = np.where(valid, 0, self.missing + 1)
            long_gone = ~valid & (self.missing >= self.HOLE_FRAMES)
            farness = np.where(valid, reading, np.where(long_gone, 255.0, self.last))
            moved = np.abs(farness - self.last) > self.STILL
            self.still = np.where(moved, 0, self.still + 1)
            self.last = farness

            person = valid & ((self.far - farness) > self.MARGIN)
            self.empty_run = self.empty_run + 1 if person.mean() < self.EMPTY_FRACTION else 0
            if self.empty_run >= self.EMPTY_FRAMES:
                self.far = farness.copy()
            else:
                revealed = valid & (reading > self.far)
                self.far[revealed] = reading[revealed]
                settled = valid & (self.still >= self.SETTLE_FRAMES)
                self.far[settled] = farness[settled]
            out.append(_clean_mask(person.astype(np.uint8) * 255))
        return out


def mask_from_relative_depth(planes):
    """Camera-only fallback: the near part of an estimated depth map."""
    out = []
    for d in planes:
        u8 = (np.clip(d, 0, 1) * 255).astype(np.uint8)
        thresh, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        person = (u8 > max(thresh, 90)).astype(np.uint8) * 255
        out.append(_clean_mask(person))
    return out


def _clean_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # Drop small islands: noise, a stray hand of someone off to the side.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        keep = np.zeros(count, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= mask.size * 0.004
        mask = (keep[labels] * 255).astype(np.uint8)
    return cv2.GaussianBlur(mask, (7, 7), 0)


# --- the world ------------------------------------------------------------------
WORLD_STYLE = ("vast dreamlike landscape seen through an archway, deep perspective, "
               "no people, " + dreamwave.STYLE)
OBJECT_STYLE = ("a single {subject}, one isolated object floating in the centre, on a "
                "pure black background, glowing, vintage paper collage cut-out")
AMBIENT_OBJECTS = ("vintage radio", "golden antique key", "paper moon", "black crow",
                   "disco mirror ball", "red telephone", "hot air balloon",
                   "glowing jellyfish", "cassette tape", "all-seeing eye")


class Viewer:
    """Where the person looking in is, smoothed: drives the parallax."""

    def __init__(self):
        self.x = 0.0        # -1 left .. 1 right, as seen on screen
        self.z = 0.0        # 0 far .. 1 close to the mirror
        self.person_z = 0.5

    def follow(self, presence, now, dt):
        if presence is not None and presence.present:
            tx = (presence.centre_x - 0.5) * 2.0
            tz = min(1.0, max(0.0, (presence.nearness - 0.25) / 0.6))
        else:
            # Nobody there: a slow drift, so the world still breathes.
            tx = 0.6 * math.sin(now * 0.13)
            tz = 0.25 + 0.2 * math.sin(now * 0.07)
        k = 1.0 - math.exp(-dt * 2.5)
        self.x += (tx - self.x) * k
        self.z += (tz - self.z) * k
        if presence is not None:
            self.person_z = 0.35 + 0.6 * presence.nearness


class FloatingObject:
    def __init__(self, subject, rgba, rng, now, spoken):
        self.subject = subject
        self.rgba = rgba
        self.spoken = spoken
        self.born = now
        # Spoken things arrive near the front, beside whoever asked rather than
        # over them; ambient things drift further back and further out.
        side = rng.choice((-1, 1))
        self.x = side * (rng.uniform(0.3, 0.5) if spoken else rng.uniform(0.4, 0.8))
        self.y = rng.uniform(-0.32, -0.08)
        self.z = rng.uniform(0.5, 0.9) if spoken else rng.uniform(0.15, 0.55)
        self.phase = rng.uniform(0, math.tau)
        self.spin = rng.uniform(-6, 6)

    def opacity(self, now):
        age = now - self.born
        return max(0.0, min(1.0, age / 1.2, (OBJECT_LIFETIME - age) / 4.0))


def object_cutout(rgb):
    """A generated object on black -> RGBA cut-out, trimmed to its bounds."""
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    alpha = np.clip((lum - 22.0) / 38.0, 0, 1)
    hard = (alpha > 0.35).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if count <= 1:
        return None
    # The biggest piece is the object; the model sometimes adds sparkles.
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < rgb.shape[0] * rgb.shape[1] * 0.02:
        return None
    region = cv2.dilate((labels == biggest).astype(np.uint8), np.ones((15, 15), np.uint8))
    alpha *= region
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    ys, xs = np.nonzero(alpha > 0.02)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])[y0:y1, x0:x1]
    return np.ascontiguousarray(rgba)


class DreamWorld:
    """The generated place, the things floating in it, and the compositing."""

    def __init__(self, submit, generate, estimate_depth, rng=None):
        # submit(job) queues job(pipe, encoder, spec) on the inference thread;
        # generate(...) -> RGB uint8; estimate_depth([rgb]) -> [plane 0..1].
        self.submit = submit
        self.generate = generate
        self.estimate_depth = estimate_depth
        self.rng = rng or random.Random()
        self.lock = threading.Lock()
        self.backdrop = None        # (rgb, depth) at native size
        self.previous = None
        self.changed_at = 0.0
        self.objects = []
        self.viewer = Viewer()
        self.pending_backdrop = None
        self._grid_key = None
        self._grid = None
        self._work = {}             # per-backdrop resized copies
        self._last_ambient = 0.0
        self.active = False

    # -- requests (any thread) ---------------------------------------------
    def request_backdrop(self, place, seed=None):
        prompt = f"{place}, {WORLD_STYLE}"
        seed = self.rng.randrange(1, 10_000) if seed is None else seed
        token = object()
        self.pending_backdrop = token

        def job(pipe, encoder, spec):
            if self.pending_backdrop is not token:
                return                          # superseded before it ran
            rgb = self.generate(pipe, encoder, spec, prompt, seed)
            depth = self.estimate_depth([rgb])[0].astype(np.float32)
            depth = cv2.GaussianBlur(depth, (0, 0), 3)
            with self.lock:
                self.previous = self.backdrop
                self.backdrop = (rgb, depth)
                self.changed_at = time.time()
            print(f"world: built {place[:60]!r}", flush=True)
        self.submit(job)

    def request_object(self, subject, spoken=True, seed=None):
        prompt = OBJECT_STYLE.format(subject=subject)
        seed = self.rng.randrange(1, 10_000) if seed is None else seed

        def job(pipe, encoder, spec):
            rgb = self.generate(pipe, encoder, spec, prompt, seed)
            rgba = object_cutout(rgb)
            if rgba is None:
                print(f"world: could not cut out {subject!r}", flush=True)
                return
            with self.lock:
                now = time.time()
                self.objects.append(FloatingObject(subject, rgba, self.rng, now, spoken))
                # Oldest ambient things make way first, then the oldest of all.
                while len(self.objects) > MAX_OBJECTS:
                    ambient = [o for o in self.objects if not o.spoken]
                    self.objects.remove(ambient[0] if ambient else self.objects[0])
            print(f"world: {subject!r} is floating in the world", flush=True)
        self.submit(job)

    # -- per frame (render thread) ------------------------------------------
    def update(self, now, dt, presence):
        self.viewer.follow(presence, now, dt)
        with self.lock:
            self.objects = [o for o in self.objects if o.opacity(now) > 0
                            or now - o.born < 1.0]
            few = len(self.objects) < 3
        if self.active and few and now - self._last_ambient > 25.0 \
                and self.backdrop is not None:
            self._last_ambient = now
            self.request_object(self.rng.choice(AMBIENT_OBJECTS), spoken=False)

    def ready(self):
        return self.backdrop is not None

    def _work_copy(self, item, ww, wh):
        key = (id(item), ww, wh)
        got = self._work.get(key)
        if got is None:
            rgb, depth = item
            # Overscan, so parallax never pulls an edge into view.
            img = cv2.resize(rgb[:, :, ::-1], (ww, wh), interpolation=cv2.INTER_CUBIC)
            dep = cv2.resize(depth, (ww, wh), interpolation=cv2.INTER_LINEAR)
            got = (img, dep)
            if len(self._work) > 4:
                self._work.clear()
            self._work[key] = got
        return got

    def _parallax(self, item, ww, wh):
        img, dep = self._work_copy(item, ww, wh)
        if self._grid_key != (ww, wh):
            xs, ys = np.meshgrid(np.arange(ww, dtype=np.float32),
                                 np.arange(wh, dtype=np.float32))
            self._grid = (xs, ys)
            self._grid_key = (ww, wh)
        xs, ys = self._grid
        v = self.viewer
        # Nearer pixels move further, and stepping in pushes them past you.
        near = dep - 0.35
        zoom = 1.0 / (1.10 + DOLLY * v.z * (0.4 + dep))
        cx, cy = ww / 2.0, wh * 0.55
        map_x = cx + (xs - cx) * zoom - v.x * PARALLAX * ww * near
        map_y = cy + (ys - cy) * zoom - 0.02 * wh * near
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    def _draw_objects(self, canvas, now, front, person_z):
        h, w = canvas.shape[:2]
        v = self.viewer
        with self.lock:
            objects = list(self.objects)
        for o in sorted(objects, key=lambda o: o.z):
            if (o.z > person_z) != front:
                continue
            alpha = o.opacity(now)
            if alpha <= 0:
                continue
            size = w * (0.12 + 0.2 * o.z) * (1.0 + DOLLY * v.z * o.z)
            oh, ow = o.rgba.shape[:2]
            scale = size / max(oh, ow)
            bob = 0.025 * math.sin(now * 0.8 + o.phase)
            cx = w / 2 + (o.x * 0.5 - v.x * PARALLAX * 1.6 * (o.z - 0.35)) * w
            cy = h * 0.5 + (o.y + bob) * h
            ang = o.spin * math.sin(now * 0.3 + o.phase)
            m = cv2.getRotationMatrix2D((ow / 2, oh / 2), ang, scale)
            out_w, out_h = int(ow * scale * 1.2) + 2, int(oh * scale * 1.2) + 2
            m[0, 2] += out_w / 2 - ow / 2
            m[1, 2] += out_h / 2 - oh / 2
            sprite = cv2.warpAffine(o.rgba, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            dreamwave.blit_rgba(canvas, sprite, int(cx - out_w / 2), int(cy - out_h / 2),
                                opacity=alpha)

    def render(self, person_bgr, person_alpha, now):
        """The world at stage size with the person composited into it."""
        h, w = person_bgr.shape[:2]
        ww, wh = max(64, int(w * WORK_SCALE)), max(64, int(h * WORK_SCALE))
        with self.lock:
            current, previous, changed = self.backdrop, self.previous, self.changed_at
        world = self._parallax(current, ww, wh)
        k = (now - changed) / BACKDROP_FADE
        if previous is not None and k < 1.0:
            world = cv2.addWeighted(self._parallax(previous, ww, wh), 1.0 - k, world, k, 0)
        world = cv2.resize(world, (w, h), interpolation=cv2.INTER_LINEAR)

        person_z = self.viewer.person_z
        self._draw_objects(world, now, front=False, person_z=person_z)
        if person_alpha is not None:
            a = person_alpha.astype(np.float32) * (1.0 / 255.0)
            # A faint gold rim where you meet the world, as if stepping through.
            rim = cv2.GaussianBlur(a, (0, 0), max(1.5, w * 0.004)) - a
            rim = np.clip(rim * 2.2, 0, 0.8)
            world = cv2.blendLinear(person_bgr, world, a, 1.0 - a)
            gold = np.empty_like(world)
            gold[:] = dreamwave.bgr(dreamwave.GOLD)
            world = cv2.blendLinear(gold, world, rim, 1.0 - rim)
        else:
            world = person_bgr
        self._draw_objects(world, now, front=True, person_z=person_z)
        return world
