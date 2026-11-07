"""Screen composition for the local mirror: 16:9 stage, spinning coin, and the
transcript text that scrolls along wavy paths.

Everything here mirrors what the React frontend (`web/src/components/WarpPage.tsx`)
draws, in the same coordinate spaces, so the local window and the hosted page
look the same.
"""

import math
import random
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from svg_logo import load_mischief_alpha

# --- stage -----------------------------------------------------------------
STAGE_ASPECT = 16.0 / 9.0

# --- coin (WarpPage: COIN_RELATIVE_SIZE / ROTATION_DURATION / coin-rotate) --
COIN_RELATIVE_SIZE = 0.05      # fraction of stage width, 16:9 layout
COIN_ROTATION_SECONDS = 3.0
COIN_BOTTOM_FRACTION = 0.04    # `bottom: 4%` of stage height
COIN_TILT_DEGREES = 20.0       # rotateX(20deg)
COIN_PERSPECTIVE_PX = 1000.0   # perspective: 1000px
COIN_BLUE_RGB = (34, 161, 236)    # #22A1EC
COIN_PINK_RGB = (255, 186, 241)   # #FFBAF1

# --- transcripts (WarpPage: SCROLL_DURATION / PATH_* / the two <textPath>s) -
SCROLL_DURATION = 5.0
VIEWBOX_W = 900.0
VIEWBOX_H = 400.0
PATH_BASE_Y = 350.0
PATH_AMPLITUDE = 50.0
# `text-lg` on the <text>, in viewBox units; the square build uses `text-3xl`.
TEXT_SIZE_VB = 18.0


# Each transcript gets its own route across the screen rather than one of two
# fixed curves. Seeded from the transcript itself, so a line keeps the same path
# for its whole life while no two lines share one.
PATH_BAND_TOP = 0.10        # fraction of the band's height a path may start at
PATH_BAND_BOTTOM = 0.84
PATH_WAVES = (2, 5)         # inclusive range of crests
PATH_AMPLITUDE_RANGE = (45.0, 120.0)
PATH_TILT = 55.0            # how far the two ends may differ in height
PATH_MAX_ELEVATION = 45.0   # degrees from horizontal an approach may take


# Lines arriving close together are put in separate lanes, so two transcripts on
# screen at once do not land on top of each other.
PATH_LANES = 4


def _catmull_rom_chain(points):
    """Smooth cubic chain through `points`, as `M p0 C ...` control values.

    Building each wave as its own bezier and alternating the control offsets up
    and down leaves the tangent reversing at every joint, so the curve arrives
    at a waypoint going one way and leaves going the other — a run of scallops
    rather than a flowing line. Catmull-Rom sets each waypoint's tangent from
    its neighbours instead, which is continuous across the joints by
    construction.
    """
    if len(points) < 2:
        return list(points)
    padded = [points[0]] + list(points) + [points[-1]]
    chain = [points[0]]
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        chain.append((p1[0] + (p2[0] - p0[0]) / 6.0,
                      p1[1] + (p2[1] - p0[1]) / 6.0))
        chain.append((p2[0] - (p3[0] - p1[0]) / 6.0,
                      p2[1] - (p3[1] - p1[1]) / 6.0))
        chain.append(p2)
    return chain


def converging_path(seed, lane, lanes, width, height, target, coin_radius):
    """A path that comes in from off-screen and ends at the cat.

    Every line travels the same way: in from an edge and down into the coin,
    which is what makes the mirror look like it is listening rather than like
    text is being scrolled past it. Lines arriving together are given different
    approach directions so they converge from around the room rather than
    stacking on one another.
    """
    rng = random.Random(seed)

    # Approaches stay within PATH_MAX_ELEVATION of horizontal and always come
    # from above the coin, which sits near the bottom of the frame. A steeper
    # line has almost no screen to travel down and arrives before it can be
    # read. Alternate lanes take opposite sides, so consecutive lines come in
    # from opposite directions rather than always the same one.
    lanes = max(2, lanes)
    lane = lane % lanes
    from_right = bool(lane % 2)
    per_side = lanes // 2
    sub = lane // 2
    wedge = math.radians(PATH_MAX_ELEVATION)
    elevation = wedge * (sub + rng.uniform(0.12, 0.88)) / per_side
    # Screen y grows downward, so a negative sine puts the start above the coin.
    angle = -elevation if from_right else math.pi + elevation

    # Start well outside the frame so a line is already moving when it appears.
    reach = math.hypot(width, height) * rng.uniform(0.62, 0.78)
    start = (target[0] + math.cos(angle) * reach,
             target[1] + math.sin(angle) * reach)

    # Stop short of the centre: the glyphs should be gone by the time they would
    # overlap the coin, and the fade does the rest.
    end_gap = coin_radius * 0.55
    end = (target[0] - math.cos(angle) * end_gap,
           target[1] - math.sin(angle) * end_gap)

    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / distance, dx / distance      # perpendicular to the approach

    waves = rng.randint(*PATH_WAVES)
    swing = distance * rng.uniform(0.05, 0.11)
    direction = 1 if rng.random() < 0.5 else -1

    waypoints = [start]
    for i in range(1, waves):
        t = i / waves
        # The wobble tapers to nothing at the cat, so the line settles into it
        # instead of thrashing about as it arrives.
        taper = (1.0 - t) ** 1.4
        offset = swing * direction * rng.uniform(0.75, 1.25) * taper
        waypoints.append((start[0] + dx * t + nx * offset,
                          start[1] + dy * t + ny * offset))
        direction = -direction
    waypoints.append(end)
    return _catmull_rom_chain(waypoints)


def _wavy_paths(base_y, amplitude):
    """The frontend's two <path> definitions at a given height and amplitude."""
    left = [
        (0, base_y),
        (100, base_y + amplitude), (200, base_y - amplitude), (300, base_y),
        (350, base_y + amplitude), (400, base_y - amplitude), (450, base_y),
    ]
    right = [
        (450, base_y),
        (500, base_y - amplitude), (550, base_y + amplitude), (600, base_y),
        (650, base_y + amplitude), (700, base_y - amplitude), (900, base_y),
    ]
    return left, right


class Layout:
    """Overlay geometry, which the frontend retunes per stage shape.

    The 16:9 build puts the paths at y=350 with a 50 unit wave and sizes the
    coin at 5% of the stage; the square `mischief 3.0` build raises the paths to
    300, widens the wave to 60 and grows the coin to 9%, both commented there as
    adjusted for the square layout.
    """

    def __init__(self, coin_relative_size, path_base_y, path_amplitude,
                 text_size_vb=TEXT_SIZE_VB):
        self.coin_relative_size = coin_relative_size
        self.path_base_y = path_base_y
        self.path_amplitude = path_amplitude
        self.text_size_vb = text_size_vb
        left, right = _wavy_paths(path_base_y, path_amplitude)
        self.left_path = _Path(left)
        self.right_path = _Path(right)


def layout_for(stage_aspect):
    """Square-ish stages get the square build's geometry, wide ones the 16:9."""
    if stage_aspect < 1.2:
        return Layout(0.09, 300.0, 60.0, 44.0)      # sized for the full-width sweep
    # Wide stages keep the square's text size (glyphs scale with stage height,
    # so it comes out the same on screen) and a coin of about the same pixel
    # size: the hosted build's 18 unit text and 5% coin were tuned for text
    # confined to the picture, not sweeping in across the whole screen.
    return Layout(0.055, PATH_BASE_Y, PATH_AMPLITUDE, 44.0)
LEFT_PATH = [
    (0, PATH_BASE_Y),
    (100, PATH_BASE_Y + PATH_AMPLITUDE), (200, PATH_BASE_Y - PATH_AMPLITUDE), (300, PATH_BASE_Y),
    (350, PATH_BASE_Y + PATH_AMPLITUDE), (400, PATH_BASE_Y - PATH_AMPLITUDE), (450, PATH_BASE_Y),
]
RIGHT_PATH = [
    (450, PATH_BASE_Y),
    (500, PATH_BASE_Y - PATH_AMPLITUDE), (550, PATH_BASE_Y + PATH_AMPLITUDE), (600, PATH_BASE_Y),
    (650, PATH_BASE_Y + PATH_AMPLITUDE), (700, PATH_BASE_Y - PATH_AMPLITUDE), (900, PATH_BASE_Y),
]
# leftFade runs opaque->transparent over x 0..40% of the 900-wide mask rect;
# rightFade mirrors it from the right edge. Only used by the fixed-path layout.
FADE_SPAN_VB = 0.40 * VIEWBOX_W
# A line now fades up as it enters and away as it leaves, over this much of its
# journey, rather than being masked toward the middle of the screen.
TRAVEL_FADE = 0.18
# Lines are tinted from the piece's own palette, picked per transcript.
TEXT_COLOURS = ((255, 255, 255), COIN_BLUE_RGB, COIN_PINK_RGB)
# Text dissolves as it reaches the coin rather than sliding under it: fully
# clear at this multiple of the coin's radius, untouched beyond the outer one.
COIN_FADE_INNER = 1.15
COIN_FADE_OUTER = 2.6


def coin_centre(stage_rect, relative_size=COIN_RELATIVE_SIZE):
    """Where the coin sits, as (cx, cy, size), for whatever needs to avoid it."""
    x, y, w, h = stage_rect
    size = max(8, int(round(w * relative_size)))
    return x + w / 2.0, y + h - h * COIN_BOTTOM_FRACTION - size / 2.0, size
# The frontend puts the transcript band in the bottom 40% of its content box.
BAND_FRACTION = 0.40


def stage_rect(screen_w, screen_h, aspect=STAGE_ASPECT):
    """Largest stage of `aspect` that fits the display, centred (letterboxed).

    `aspect=None` fills the display instead, leaving no bars.
    """
    if aspect is None:
        return 0, 0, screen_w, screen_h
    if screen_w / screen_h > aspect:
        h = screen_h
        w = int(round(h * aspect))
    else:
        w = screen_w
        h = int(round(w / aspect))
    return (screen_w - w) // 2, (screen_h - h) // 2, w, h


class MorphingCircles:
    """The animated background from the `mischief 3.0` frontend.

    A 135-degree base gradient with ten heavily blurred circles drifting over
    it, bouncing off the edges and pulsing in size. Ported from that build's
    `PageContainer` and its circle simulation, keeping the same colours, sizes,
    speeds, opacity and 50ms tick.

    The circles are simulated and drawn at a fraction of the screen resolution
    and scaled up. An 80px blur destroys any detail finer than that anyway, so
    the result is the same and a frame costs well under a millisecond.
    """

    # linear-gradient(135deg, #1a8cd8 0%, #22A1EC 25%, #FFBAF1 75%, #e89dd9 100%)
    BASE_ANGLE_DEGREES = 135.0
    BASE_STOPS = (
        (0.00, (0x1a, 0x8c, 0xd8)),
        (0.25, (0x22, 0xA1, 0xEC)),
        (0.75, (0xFF, 0xBA, 0xF1)),
        (1.00, (0xe8, 0x9d, 0xd9)),
    )
    CIRCLE_COLORS = ((0x22, 0xA1, 0xEC), (0xFF, 0xBA, 0xF1))
    COUNT = 10
    OPACITY = 0.7
    BLUR_PX = 80.0          # CSS filter: blur(80px) == stdDeviation 80
    TICK_SECONDS = 0.05     # setInterval(..., 50)
    POSITION_LIMIT = 20.0   # bounces at -20%..120%
    SIZE_MIN = 20.0
    SIZE_MAX = 60.0
    RENDER_SCALE = 8        # simulate and draw at 1/8 size

    def __init__(self, seed=None):
        rng = random.Random(seed)
        self.circles = [
            {
                "x": rng.random() * 100.0,
                "y": rng.random() * 100.0,
                "size": rng.random() * 30.0 + 20.0,          # 20-50 vh
                "color": self.CIRCLE_COLORS[i % 2],
                "vx": (rng.random() - 0.5) * 0.05,
                "vy": (rng.random() - 0.5) * 0.05,
                "size_vel": (rng.random() - 0.5) * 0.1,
            }
            for i in range(self.COUNT)
        ]
        self._last_tick = None
        self._base = None
        self._base_size = None
        self._rendered = None
        self._rendered_size = None

    def _step(self):
        """One 50ms tick of the frontend's circle simulation."""
        limit = self.POSITION_LIMIT
        for c in self.circles:
            x = c["x"] + c["vx"]
            y = c["y"] + c["vy"]
            size = c["size"] + c["size_vel"]
            if x < -limit or x > 100.0 + limit:
                c["vx"] = -c["vx"]
                x = -limit if x < -limit else 100.0 + limit
            if y < -limit or y > 100.0 + limit:
                c["vy"] = -c["vy"]
                y = -limit if y < -limit else 100.0 + limit
            if size < self.SIZE_MIN or size > self.SIZE_MAX:
                c["size_vel"] = -c["size_vel"]
                size = self.SIZE_MIN if size < self.SIZE_MIN else self.SIZE_MAX
            c["x"], c["y"], c["size"] = x, y, size

    def update(self, now):
        """Advance the simulation to `now`, in whole ticks."""
        if self._last_tick is None:
            self._last_tick = now
            return
        elapsed = now - self._last_tick
        # Cap the catch-up so a stall does not spin through thousands of ticks.
        ticks = min(int(elapsed / self.TICK_SECONDS), 40)
        if not ticks:
            return
        for _ in range(ticks):
            self._step()
        self._last_tick += ticks * self.TICK_SECONDS
        self._rendered = None   # the frame is stale now

    def _base_gradient(self, width, height):
        """The 135-degree multi-stop gradient, cached per size."""
        if self._base_size == (width, height):
            return self._base
        angle = math.radians(self.BASE_ANGLE_DEGREES)
        sin_a, cos_a = math.sin(angle), math.cos(angle)
        # CSS measures the gradient line from the centre; this is its length.
        length = abs(width * sin_a) + abs(height * cos_a)
        ys = (np.arange(height, dtype=np.float32) - height / 2.0)[:, None]
        xs = (np.arange(width, dtype=np.float32) - width / 2.0)[None, :]
        t = np.clip((xs * sin_a - ys * cos_a) / length + 0.5, 0.0, 1.0)

        stops = self.BASE_STOPS
        out = np.empty((height, width, 3), dtype=np.float32)
        positions = np.array([p for p, _ in stops], dtype=np.float32)
        for channel in range(3):
            # BGR out of RGB stops
            values = np.array([c[2 - channel] for _, c in stops], dtype=np.float32)
            out[:, :, channel] = np.interp(t, positions, values)
        self._base = out.astype(np.uint8)
        self._base_size = (width, height)
        return self._base

    def render(self, width, height):
        """Full-size BGR background for the current simulation state.

        Cached between simulation ticks: the state only changes 20 times a
        second, so there is nothing to redraw on most display frames.
        """
        if self._rendered is not None and self._rendered_size == (width, height):
            return self._rendered
        small_w = max(16, width // self.RENDER_SCALE)
        small_h = max(16, height // self.RENDER_SCALE)
        scale = small_h / float(height)

        layer = self._base_gradient(small_w, small_h).astype(np.float32)
        sigma = max(0.5, self.BLUR_PX * scale)
        ksize = int(sigma * 4) | 1

        for c in self.circles:
            # size is in vh, position in percent of the viewport
            radius = c["size"] / 100.0 * small_h / 2.0
            cx = c["x"] / 100.0 * small_w
            cy = c["y"] / 100.0 * small_h
            if radius < 0.5:
                continue
            alpha = np.zeros((small_h, small_w), dtype=np.float32)
            cv2.circle(alpha, (int(round(cx)), int(round(cy))),
                       int(round(radius)), 1.0, -1, lineType=cv2.LINE_AA)
            # Each circle is its own blurred element in the DOM, so each is
            # blurred on its own before being composited.
            alpha = cv2.GaussianBlur(alpha, (ksize, ksize), sigma)
            alpha *= self.OPACITY
            color = np.array((c["color"][2], c["color"][1], c["color"][0]),
                             dtype=np.float32)
            layer += (color - layer) * alpha[:, :, None]

        small = np.clip(layer, 0, 255).astype(np.uint8)
        self._rendered = cv2.resize(small, (width, height),
                                    interpolation=cv2.INTER_LINEAR)
        self._rendered_size = (width, height)
        return self._rendered


def cover_fit(frame_bgr, width, height):
    """Scale to fill `width` x `height`, preserving aspect, cropping overflow.

    Same result as the frontend's drawImage() cover maths, but the crop is taken
    before the resize so only the visible pixels are resampled.
    """
    fh, fw = frame_bgr.shape[:2]
    scale = max(width / fw, height / fh)
    src_w = min(fw, max(1, int(round(width / scale))))
    src_h = min(fh, max(1, int(round(height / scale))))
    x0 = (fw - src_w) // 2
    y0 = (fh - src_h) // 2
    cropped = frame_bgr[y0:y0 + src_h, x0:x0 + src_w]
    if (src_w, src_h) == (width, height):
        # Always a fresh buffer: callers draw overlays into the result, and the
        # source frame is still owned by the frame queue.
        return cropped.copy()
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(cropped, (width, height), interpolation=interp)


def rotate_stage(image_bgr, degrees):
    """Rotate the stage bitmap about its centre and clip to the stage.

    Matches `transform: rotate(Ndeg)` on the canvas element, including the black
    wedges that appear at 90/270 on a non-square stage.
    """
    degrees = int(degrees) % 360
    if degrees == 0:
        return image_bgr
    if degrees == 180:
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    h, w = image_bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -degrees, 1.0)
    return cv2.warpAffine(image_bgr, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def alpha_composite(base_bgr, overlay_bgr, overlay_alpha, x=0, y=0):
    """Composite a BGR layer with a float [0,1] alpha onto `base_bgr` in place."""
    H, W = base_bgr.shape[:2]
    h, w = overlay_alpha.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)
    a = overlay_alpha[oy0:oy1, ox0:ox1]
    if a.ndim == 2:
        a = a[:, :, None]
    region = base_bgr[y0:y1, x0:x1].astype(np.float32)
    src = overlay_bgr[oy0:oy1, ox0:ox1].astype(np.float32)
    base_bgr[y0:y1, x0:x1] = np.clip(region * (1.0 - a) + src * a, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Coin
# ---------------------------------------------------------------------------

def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


class CoinRenderer:
    """The mischief logo as a coin spinning about Y with a fixed X tilt.

    Reproduces the frontend's `rotateY(t) rotateX(20deg)` under a 1000px
    perspective, with `backface-visibility: hidden` picking which tinted face
    is showing.
    """

    def __init__(self, assets_dir, blue_rgb=COIN_BLUE_RGB, pink_rgb=COIN_PINK_RGB,
                 relative_size=COIN_RELATIVE_SIZE, faces=None):
        self.assets_dir = Path(assets_dir)
        # Optional `faces(size) -> {"front": (bgr, alpha), "back": ...}` for a
        # coin with artwork of its own rather than the tinted mischief logo.
        self.face_factory = faces
        self.relative_size = relative_size
        self.blue_rgb = blue_rgb
        self.pink_rgb = pink_rgb
        self._size = 0
        self._faces = {}

    def _ensure_size(self, size):
        if size == self._size:
            return
        if self.face_factory is not None:
            made = self.face_factory(size)
            self._faces = {"blue": made["front"], "pink": made["back"]}
            self._size = size
            return
        alpha = load_mischief_alpha(self.assets_dir, size).astype(np.float32) / 255.0
        self._faces = {}
        for name, rgb in (("blue", self.blue_rgb), ("pink", self.pink_rgb)):
            bgr = np.zeros((size, size, 3), dtype=np.uint8)
            bgr[:, :] = (rgb[2], rgb[1], rgb[0])
            self._faces[name] = (bgr, alpha)
        self._size = size

    def draw(self, surface_bgr, now, rect=None):
        """Draw the coin at its frontend position and size.

        `rect` is the stage within `surface_bgr`; the coin is sized and placed
        against the stage even when it is painted onto the whole screen, so that
        it can sit above text that sweeps outside the picture.
        """
        if rect is None:
            H, W = surface_bgr.shape[:2]
            origin_x, origin_y = 0, 0
        else:
            origin_x, origin_y, W, H = rect
        size = max(8, int(round(W * self.relative_size)))
        self._ensure_size(size)

        theta = (now % COIN_ROTATION_SECONDS) / COIN_ROTATION_SECONDS * 2.0 * math.pi
        tilt = math.radians(COIN_TILT_DEGREES)
        base = _rot_y(theta) @ _rot_x(tilt)

        showing_front = math.cos(theta) > 0.0
        if showing_front:
            R = base
            face_bgr, face_alpha = self._faces["blue"]
        else:
            R = base @ _rot_y(math.pi)
            face_bgr, face_alpha = self._faces["pink"]

        # Centre of the coin box: horizontally centred, `bottom: 4%`.
        cx = origin_x + W / 2.0
        cy = origin_y + H - H * COIN_BOTTOM_FRACTION - size / 2.0

        half = size / 2.0
        local = np.array([
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ], dtype=np.float64)
        projected = []
        for p in local:
            q = R @ p
            w = 1.0 - q[2] / COIN_PERSPECTIVE_PX
            if w <= 1e-6:
                return  # degenerate; skip this frame rather than blow up
            projected.append((cx + q[0] / w, cy + q[1] / w))
        dst = np.array(projected, dtype=np.float32)

        # Edge-on: nothing to draw.
        if cv2.contourArea(dst) < 1.0:
            return

        x_min = int(math.floor(dst[:, 0].min()))
        y_min = int(math.floor(dst[:, 1].min()))
        x_max = int(math.ceil(dst[:, 0].max()))
        y_max = int(math.ceil(dst[:, 1].max()))
        out_w, out_h = x_max - x_min, y_max - y_min
        if out_w <= 0 or out_h <= 0:
            return

        src = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(
            src, (dst - (x_min, y_min)).astype(np.float32))
        warped_bgr = cv2.warpPerspective(face_bgr, m, (out_w, out_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
        warped_alpha = cv2.warpPerspective(face_alpha, m, (out_w, out_h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha_composite(surface_bgr, warped_bgr, np.clip(warped_alpha, 0.0, 1.0),
                        x_min, y_min)


# ---------------------------------------------------------------------------
# Transcript text on curved paths
# ---------------------------------------------------------------------------

def _flatten_bezier_chain(points, segments_per_curve=60):
    """Flatten `M p0 C p1 p2 p3 C p4 p5 p6 ...` into a polyline."""
    poly = [points[0]]
    for i in range(1, len(points) - 2, 3):
        p0 = poly[-1]
        p1, p2, p3 = points[i], points[i + 1], points[i + 2]
        for k in range(1, segments_per_curve + 1):
            t = k / segments_per_curve
            u = 1.0 - t
            x = (u ** 3 * p0[0] + 3 * u * u * t * p1[0]
                 + 3 * u * t * t * p2[0] + t ** 3 * p3[0])
            y = (u ** 3 * p0[1] + 3 * u * u * t * p1[1]
                 + 3 * u * t * t * p2[1] + t ** 3 * p3[1])
            poly.append((x, y))
    return poly


class _Path:
    """Arc-length parameterised polyline."""

    @classmethod
    def from_polyline(cls, points):
        path = cls.__new__(cls)
        path._init_from(np.asarray(points, dtype=np.float64))
        return path

    def __init__(self, control_points):
        self._init_from(np.array(_flatten_bezier_chain(control_points),
                                 dtype=np.float64))

    def _init_from(self, poly):
        deltas = np.diff(poly, axis=0)
        seg_len = np.hypot(deltas[:, 0], deltas[:, 1])
        self.points = poly
        self.cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.length = float(self.cumulative[-1])

    def at(self, s):
        """Return (x, y, tangent_angle_radians) at arc length `s`."""
        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.cumulative, s, side="right")) - 1
        i = min(max(i, 0), len(self.points) - 2)
        span = self.cumulative[i + 1] - self.cumulative[i]
        t = 0.0 if span <= 0 else (s - self.cumulative[i]) / span
        p0, p1 = self.points[i], self.points[i + 1]
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        return x, y, math.atan2(p1[1] - p0[1], p1[0] - p0[0])


_LEFT = _Path(LEFT_PATH)
_RIGHT = _Path(RIGHT_PATH)


def find_font(explicit=None, assets_dir=None):
    """Locate Sigmar (the frontend's transcript face), else a system fallback."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if assets_dir:
        candidates.append(Path(assets_dir) / "Sigmar-Regular.ttf")
    candidates += [
        Path("/System/Library/Fonts/Supplemental/Impact.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]
    for path in candidates:
        if path and path.exists():
            return str(path)
    return None


# --- idle call to action ----------------------------------------------------
IDLE_AFTER_SECONDS = 40.0      # quiet for this long and the mirror asks
IDLE_FADE_SECONDS = 3.5        # deliberately languid, in and out
IDLE_VISIBLE_SECONDS = 30.0    # how long one appearance lasts, fades included
IDLE_REPEAT_SECONDS = 300.0    # and how long before it may appear again
IDLE_HEADLINE = "pick up the phone"
IDLE_SUGGESTIONS = (
    "and tell me what to become",
    "ask for something ridiculous",
    "try: turn everyone into cats",
    "whisper something you'd regret",
    "go on, I'm listening",
    "say it like you mean it",
)


class IdlePrompt:
    """The centred nudge shown when nobody has said anything for a while.

    Drawn with PIL for the same reason the settings panel is: real type and a
    soft shadow, rather than stroked polygons.
    """

    def __init__(self, font_path=None, assets_dir=None,
                 headline=IDLE_HEADLINE, suggestions=IDLE_SUGGESTIONS,
                 after=IDLE_AFTER_SECONDS, accent=COIN_PINK_RGB,
                 headline_colour=(255, 255, 255)):
        self.font_path = find_font(font_path, assets_dir)
        self.accent = tuple(accent)
        self.headline_colour = tuple(headline_colour)
        self.headline = headline
        self.suggestions = tuple(suggestions) or ("",)
        self.after = after
        self._fonts = {}
        self._cache = None
        self._cache_key = None
        self._show_started = None
        self._last_show = None
        # Chosen once when an appearance begins and held until it ends, so the
        # wording cannot change on its way off screen.
        self._suggestion = suggestions[0] if suggestions else ""
        self._spell = 0

    def _font(self, size):
        font = self._fonts.get(size)
        if font is None:
            font = (ImageFont.truetype(self.font_path, size) if self.font_path
                    else ImageFont.load_default())
            self._fonts[size] = font
        return font

    def visible(self, now, last_heard=None):
        """How strongly to show it, 0 to 1, easing in and out.

        An appearance runs for a fixed window once it starts — it is a nudge,
        not a nag, and it should not sit there for as long as a room happens to
        stay quiet.
        """
        if self._show_started is None:
            return 0.0
        elapsed = now - self._show_started
        if elapsed < 0 or elapsed > IDLE_VISIBLE_SECONDS:
            return 0.0
        raw = min(1.0,
                  elapsed / IDLE_FADE_SECONDS,
                  (IDLE_VISIBLE_SECONDS - elapsed) / IDLE_FADE_SECONDS)
        raw = max(0.0, raw)
        # Smoothstep, so it does not arrive and leave at a constant rate.
        return raw * raw * (3.0 - 2.0 * raw)

    def _track(self, now, last_heard):
        """Start an appearance when it is due, and end it when its time is up."""
        if self._show_started is not None:
            if now - self._show_started >= IDLE_VISIBLE_SECONDS:
                self._last_show = self._show_started
                self._show_started = None
            return

        quiet = now if last_heard is None else now - last_heard
        if quiet < self.after:
            return
        if self._last_show is not None and \
                now - self._last_show < IDLE_REPEAT_SECONDS:
            return                  # shown recently enough; leave the room alone
        self._show_started = now
        self._spell += 1
        self._suggestion = self.suggestions[self._spell % len(self.suggestions)]

    def _render_line(self, text, font, colour):
        """One line of type on its own transparent panel, trimmed to fit."""
        measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        box = measure.textbbox((0, 0), text, font=font)
        pad = max(4, int((box[3] - box[1]) * 0.22))
        size = (box[2] - box[0] + pad * 2, box[3] - box[1] + pad * 2)
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        origin = (pad - box[0], pad - box[1])
        for offset in ((0, 3), (0, 2)):
            d.text((origin[0] + offset[0], origin[1] + offset[1]), text,
                   font=font, fill=(0, 0, 0, 150))
        d.text(origin, text, font=font, fill=colour)
        return np.asarray(img)

    def _panels(self, height, suggestion):
        """The two lines, cached until the size or the wording changes."""
        key = (height, suggestion)
        if key == self._cache_key:
            return self._cache
        big = self._font(max(18, int(height * 0.085)))
        small = self._font(max(12, int(height * 0.042)))
        self._cache = (
            self._render_line(self.headline, big, self.headline_colour + (255,)),
            self._render_line(suggestion, small, self.accent + (255,)),
        )
        self._cache_key = key
        return self._cache

    def draw(self, surface_bgr, now, last_heard, rect=None):
        self._track(now, last_heard)
        strength = self.visible(now)
        if strength <= 0.0:
            return
        H, W = surface_bgr.shape[:2]
        if rect is None:
            rect = (0, 0, W, H)
        x, y, w, h = rect
        headline, suggestion = self._panels(h, self._suggestion)

        gap = int(h * 0.034)
        # Lay the lines out by their centres and from the *unwobbled* heights:
        # the warped panels carry a margin for the rotation, and spacing off
        # that pushes the second line far down the screen.
        head_h, sugg_h = headline.shape[0], suggestion.shape[0]
        centre_x = x + w // 2
        head_centre = y + int(h * 0.40)
        centres = (head_centre,
                   head_centre + head_h // 2 + sugg_h // 2 + gap)

        for index, panel in enumerate((headline, suggestion)):
            ph, pw = panel.shape[:2]
            px = centre_x - pw // 2
            py = centres[index] - ph // 2
            x0, y0 = max(0, px), max(0, py)
            x1, y1 = min(W, px + pw), min(H, py + ph)
            if x1 <= x0 or y1 <= y0:
                continue
            piece = panel[y0 - py:y1 - py, x0 - px:x1 - px]
            alpha = piece[:, :, 3:4].astype(np.float32) / 255.0 * strength
            rgb = piece[:, :, :3][:, :, ::-1].astype(np.float32)
            region = surface_bgr[y0:y1, x0:x1].astype(np.float32)
            surface_bgr[y0:y1, x0:x1] = np.clip(
                region * (1.0 - alpha) + rgb * alpha, 0, 255).astype(np.uint8)


class TranscriptRenderer:
    """Draws live transcripts as glyphs running along the two wavy paths."""

    # Rotation is quantised before caching rotated glyphs; 2 degrees is well
    # under what is visible at these sizes and keeps the cache small.
    ANGLE_STEP_DEGREES = 1.0
    MAX_CACHE = 4096

    def __init__(self, font_path=None, assets_dir=None, layout=None):
        self.layout = layout
        self.font_path = find_font(font_path, assets_dir)
        self._font = None
        self._font_big = None
        self._font_size = 0
        self._glyphs = {}
        self._rotated = {}
        self._paths = {}

    def _ensure_font(self, size_px):
        if self._font is not None and size_px == self._font_size:
            return
        if self.font_path:
            self._font = ImageFont.truetype(self.font_path, size_px)
        else:
            self._font = ImageFont.load_default()
        self._font_size = size_px
        self._font_big = None
        self._glyphs.clear()
        self._rotated.clear()

    # Glyphs are drawn and rotated at this multiple of the display size, then
    # area-averaged down. Rotating a glyph rendered at final size leaves the
    # diagonals visibly ragged; supersampling is what makes the edges clean.
    GLYPH_SUPERSAMPLE = 3

    def _glyph(self, ch):
        """Alpha tile for one character, with the baseline midpoint at its centre.

        Kept at supersampled resolution — it is rotated before being reduced.
        """
        cached = self._glyphs.get(ch)
        if cached is not None:
            return cached
        scale = self.GLYPH_SUPERSAMPLE
        advance = self._font.getlength(ch)
        tile = max(8, int(round(self._font_size * 3))) * scale
        img = Image.new("L", (tile, tile), 0)
        draw = ImageDraw.Draw(img)
        # Sigmar is already a heavy face; the synthetic bold stroke only
        # thickened the edges and made them rougher.
        draw.text((tile / 2.0 - advance * scale / 2.0, tile / 2.0), ch,
                  font=self._big_font(), fill=255, anchor="ls")
        self._glyphs[ch] = (img, advance)
        return self._glyphs[ch]

    def _big_font(self):
        if self._font_big is None:
            size = self._font_size * self.GLYPH_SUPERSAMPLE
            self._font_big = (ImageFont.truetype(self.font_path, size)
                              if self.font_path else ImageFont.load_default())
        return self._font_big

    def _rotated_glyph(self, ch, angle_deg):
        bucket = round(angle_deg / self.ANGLE_STEP_DEGREES)
        key = (ch, self._font_size, bucket)
        cached = self._rotated.get(key)
        if cached is not None:
            return cached
        img, advance = self._glyph(ch)
        # Screen y grows downwards, PIL rotates counter-clockwise.
        rotated = img.rotate(-bucket * self.ANGLE_STEP_DEGREES,
                             resample=Image.BICUBIC)
        big = np.asarray(rotated, dtype=np.uint8)
        side = max(1, big.shape[0] // self.GLYPH_SUPERSAMPLE)
        arr = cv2.resize(big, (side, side), interpolation=cv2.INTER_AREA)
        if len(self._rotated) > self.MAX_CACHE:
            self._rotated.clear()
        self._rotated[key] = (arr, advance)
        return arr, advance

    def _screen_path(self, seed, lane, width, height, target, coin_radius, geometry):
        """One transcript's route into the cat, in screen pixels, cached."""
        key = (seed, lane, geometry)
        cached = self._paths.get(key)
        if cached is not None:
            return cached
        control = converging_path(seed, lane, PATH_LANES, width, height,
                                  target, coin_radius)
        path = _Path(control)
        if len(self._paths) > 64:
            self._paths.clear()
        self._paths[key] = path
        return path

    def _draw_text_on_path(self, layer, text, path, start_offset_fraction,
                           alpha=1.0):
        """Stamp `text` along `path`, centred on the given fractional offset.

        Glyphs take the path's tangent, so on a path running right to left they
        would come out upside down and backwards. When the line is heading that
        way the characters are laid down in reverse and turned through half a
        turn, which puts the text the right way up and still reading forwards.
        """
        total_advance = sum(self._glyph(ch)[1] for ch in text)
        centre = start_offset_fraction * path.length
        s = centre - total_advance / 2.0
        lh, lw = layer.shape[:2]

        _, _, mid_angle = path.at(min(max(centre, 0.0), path.length))
        flipped = math.cos(mid_angle) < 0.0
        if flipped:
            text = text[::-1]

        for ch in text:
            _, advance = self._glyph(ch)
            mid = s + advance / 2.0
            s += advance
            if ch == " ":
                continue
            if mid < 0.0 or mid > path.length:
                continue
            px, py, angle = path.at(mid)
            if flipped:
                angle += math.pi
            arr, _ = self._rotated_glyph(ch, math.degrees(angle))
            gh, gw = arr.shape
            x0 = int(round(px - gw / 2.0))
            y0 = int(round(py - gh / 2.0))
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(lw, x0 + gw), min(lh, y0 + gh)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            patch = arr[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
            if alpha < 1.0:
                patch = (patch.astype(np.float32) * alpha).astype(np.uint8)
            target = layer[sy0:sy1, sx0:sx1]
            np.maximum(target, patch, out=target)

    def _coin_fade(self, x0, y0, shape, stage_rect, relative_size):
        """Falloff so a line dissolves as it arrives at the coin."""
        cx, cy, size = coin_centre(stage_rect, relative_size)
        radius = size / 2.0
        inner, outer = radius * COIN_FADE_INNER, radius * COIN_FADE_OUTER
        h, w = shape
        ys = (np.arange(h, dtype=np.float32) + y0 - cy)[:, None]
        xs = (np.arange(w, dtype=np.float32) + x0 - cx)[None, :]
        distance = np.sqrt(xs * xs + ys * ys)
        return np.clip((distance - inner) / max(1.0, outer - inner), 0.0, 1.0)

    def _travel_alpha(self, progress):
        """Fade a line up as it enters and away as it leaves."""
        return float(min(1.0, progress / TRAVEL_FADE,
                         (1.0 - progress) / TRAVEL_FADE))

    def draw(self, surface_bgr, transcripts, now, stage_rect=None,
             coin_relative_size=None):
        """Draw every live transcript across `surface_bgr`.

        `transcripts` is [(text, started_at, side)]. `stage_rect` is the picture
        within the surface; the band is sized and positioned against it, the way
        the frontend lays it out in the bottom of its content box, while the
        sweep itself spans the full surface. Each line takes its own route and
        its own colour, both seeded from the line so they never change under it.
        """
        if not transcripts:
            return
        H, W = surface_bgr.shape[:2]
        if stage_rect is None:
            stage_rect = (0, 0, W, H)
        stage_x, stage_y, stage_w, stage_h = stage_rect
        if coin_relative_size is None and self.layout is not None:
            coin_relative_size = self.layout.coin_relative_size

        # Glyph size still comes from the stage, so widening the screen does not
        # enlarge the text with it.
        sy = stage_h * BAND_FRACTION / VIEWBOX_H
        size_vb = self.layout.text_size_vb if self.layout is not None else TEXT_SIZE_VB
        self._ensure_font(max(8, int(round(size_vb * sy))))

        cx, cy, coin_size = coin_centre(stage_rect,
                                        coin_relative_size or COIN_RELATIVE_SIZE)
        target = (cx, cy)
        coin_radius = coin_size / 2.0
        geometry = (W, H, round(cx, 1), round(cy, 1), round(coin_radius, 1))

        # One layer per colour: glyphs merge with max() within a layer, so they
        # cannot be given individual colours inside one.
        layers = {}
        for text, started_at, _side in transcripts:
            progress = (now - started_at) / SCROLL_DURATION
            if progress < 0.0 or progress > 1.0:
                continue
            seed = f"{text}|{started_at:.3f}"
            # Lane from a hash of the line, not from its arrival time: lines
            # land about every 10s and quantising that to 2.5s buckets aliased
            # exactly onto the four lanes, so every line took the same one and
            # they all came in from the same side.
            digest = zlib.crc32(seed.encode("utf-8"))
            lane = digest % PATH_LANES
            path = self._screen_path(seed, lane, W, H, target, coin_radius,
                                     geometry)
            # Always travels the same way: in from the edge, down into the cat.
            offset = progress
            colour = TEXT_COLOURS[(digest // PATH_LANES) % len(TEXT_COLOURS)]
            layer = layers.get(colour)
            if layer is None:
                layer = layers[colour] = np.zeros((H, W), dtype=np.uint8)
            self._draw_text_on_path(layer, text, path, offset,
                                    self._travel_alpha(progress))

        for colour, layer in layers.items():
            # Only touch the rows and columns the glyphs actually landed on.
            x, y, w, h = cv2.boundingRect(layer)
            if w == 0 or h == 0:
                continue
            pad = max(2, int(round(12 * sy)))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            alpha = layer[y0:y1, x0:x1].astype(np.float32) / 255.0
            if coin_relative_size:
                alpha *= self._coin_fade(x0, y0, alpha.shape, stage_rect,
                                         coin_relative_size)
            if not alpha.any():
                continue

            # `drop-shadow-lg`: two soft dark shadows under the glyphs.
            shadow = np.zeros_like(alpha)
            for dy_vb, blur_vb, strength in ((4.0, 3.0, 0.10), (10.0, 8.0, 0.04)):
                dy = max(0, int(round(dy_vb * sy)))
                k = max(1, int(round(blur_vb * sy)) | 1)
                shifted = np.roll(alpha, dy, axis=0)
                if dy > 0:
                    shifted[:dy] = 0.0
                shadow = np.maximum(shadow,
                                    cv2.GaussianBlur(shifted, (k, k), 0) * strength)
            shadow *= (1.0 - alpha)

            sh, sw = alpha.shape
            alpha_composite(surface_bgr, np.zeros((sh, sw, 3), dtype=np.uint8),
                            shadow, x0, y0)
            tint = np.empty((sh, sw, 3), dtype=np.uint8)
            tint[:, :] = (colour[2], colour[1], colour[0])      # RGB -> BGR
            alpha_composite(surface_bgr, tint, alpha, x0, y0)
