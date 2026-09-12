"""Screen composition for the local mirror: 16:9 stage, spinning coin, and the
transcript text that scrolls along wavy paths.

Everything here mirrors what the React frontend (`web/src/components/WarpPage.tsx`)
draws, in the same coordinate spaces, so the local window and the hosted page
look the same.
"""

import math
import random
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
        return Layout(0.09, 300.0, 60.0, 30.0)      # text-3xl
    return Layout(COIN_RELATIVE_SIZE, PATH_BASE_Y, PATH_AMPLITUDE, TEXT_SIZE_VB)
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
# rightFade mirrors it from the right edge.
FADE_SPAN_VB = 0.40 * VIEWBOX_W
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
                 relative_size=COIN_RELATIVE_SIZE):
        self.assets_dir = Path(assets_dir)
        self.relative_size = relative_size
        self.blue_rgb = blue_rgb
        self.pink_rgb = pink_rgb
        self._size = 0
        self._faces = {}

    def _ensure_size(self, size):
        if size == self._size:
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


class TranscriptRenderer:
    """Draws live transcripts as glyphs running along the two wavy paths."""

    # Rotation is quantised before caching rotated glyphs; 2 degrees is well
    # under what is visible at these sizes and keeps the cache small.
    ANGLE_STEP_DEGREES = 2.0
    MAX_CACHE = 4096

    def __init__(self, font_path=None, assets_dir=None, layout=None):
        self.layout = layout
        self.font_path = find_font(font_path, assets_dir)
        self._font = None
        self._font_size = 0
        self._glyphs = {}
        self._rotated = {}
        self._paths = None
        self._paths_key = None

    def _ensure_font(self, size_px):
        if self._font is not None and size_px == self._font_size:
            return
        if self.font_path:
            self._font = ImageFont.truetype(self.font_path, size_px)
        else:
            self._font = ImageFont.load_default()
        self._font_size = size_px
        self._glyphs.clear()
        self._rotated.clear()

    def _glyph(self, ch):
        """Alpha tile for one character, with the baseline midpoint at its centre."""
        cached = self._glyphs.get(ch)
        if cached is not None:
            return cached
        size = self._font_size
        advance = self._font.getlength(ch)
        tile = max(8, int(round(size * 3)))
        img = Image.new("L", (tile, tile), 0)
        draw = ImageDraw.Draw(img)
        # `font-bold` against a single-weight face: browsers synthesise bold, so
        # approximate it with a light stroke.
        stroke = max(0, int(round(size * 0.03)))
        draw.text((tile / 2.0 - advance / 2.0, tile / 2.0), ch, font=self._font,
                  fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
        self._glyphs[ch] = (img, advance)
        return self._glyphs[ch]

    def _rotated_glyph(self, ch, angle_deg):
        bucket = round(angle_deg / self.ANGLE_STEP_DEGREES)
        key = (ch, self._font_size, bucket)
        cached = self._rotated.get(key)
        if cached is not None:
            return cached
        img, advance = self._glyph(ch)
        # Screen y grows downwards, PIL rotates counter-clockwise.
        rotated = img.rotate(-bucket * self.ANGLE_STEP_DEGREES, resample=Image.BILINEAR)
        arr = np.asarray(rotated, dtype=np.uint8)
        if len(self._rotated) > self.MAX_CACHE:
            self._rotated.clear()
        self._rotated[key] = (arr, advance)
        return arr, advance

    def _screen_paths(self, key, sx, sy, band_top):
        """The two paths mapped into screen pixels, cached.

        The horizontal scale spans the whole display so transcripts sweep in
        from the edges, while the vertical scale and the glyph size come from
        the stage — otherwise widening the sweep would blow the text up with it.
        """
        if self._paths_key == key:
            return self._paths
        out = []
        for path in (self.layout.left_path, self.layout.right_path) \
                if self.layout is not None else (_LEFT, _RIGHT):
            pts = path.points.copy()
            pts[:, 0] *= sx
            pts[:, 1] = pts[:, 1] * sy + band_top
            out.append(_Path.from_polyline(pts))
        self._paths_key = key
        self._paths = out
        return out

    def _draw_text_on_path(self, layer, text, path, start_offset_fraction):
        """Stamp `text` along `path`, centred on the given fractional offset.

        Everything here is already in screen pixels.
        """
        total_advance = sum(self._glyph(ch)[1] for ch in text)
        s = start_offset_fraction * path.length - total_advance / 2.0
        lh, lw = layer.shape[:2]

        for ch in text:
            _, advance = self._glyph(ch)
            mid = s + advance / 2.0
            s += advance
            if ch == " ":
                continue
            # SVG does not render glyphs that fall off either end of the path.
            if mid < 0.0 or mid > path.length:
                continue
            px, py, angle = path.at(mid)
            arr, _ = self._rotated_glyph(ch, math.degrees(angle))
            gh, gw = arr.shape
            x0 = int(round(px - gw / 2.0))
            y0 = int(round(py - gh / 2.0))
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(lw, x0 + gw), min(lh, y0 + gh)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            patch = arr[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
            target = layer[sy0:sy1, sx0:sx1]
            np.maximum(target, patch, out=target)

    def _fade_mask(self, width, scale, tx, is_left):
        """Alpha ramp matching the leftFade / rightFade gradients."""
        xs = (np.arange(width, dtype=np.float32) - tx) / scale  # viewBox x
        if is_left:
            ramp = 1.0 - xs / FADE_SPAN_VB
        else:
            ramp = 1.0 - (VIEWBOX_W - xs) / FADE_SPAN_VB
        return np.clip(ramp, 0.0, 1.0)

    def draw(self, surface_bgr, transcripts, now, stage_rect=None):
        """Draw every live transcript across `surface_bgr`.

        `transcripts` is [(text, started_at, side)]. `stage_rect` is the picture
        within the surface; the text band is sized and positioned against it,
        the way the frontend lays it out in the bottom 40% of its content box,
        while the sweep itself spans the full surface.
        """
        if not transcripts:
            return
        H, W = surface_bgr.shape[:2]
        if stage_rect is None:
            stage_rect = (0, 0, W, H)
        stage_x, stage_y, stage_w, stage_h = stage_rect

        # The frontend's band is the bottom 40% of the content box, with the
        # viewBox fitted into it: 0.4 * stage_h / VIEWBOX_H.
        sy = stage_h * BAND_FRACTION / VIEWBOX_H
        band_top = stage_y + stage_h * (1.0 - BAND_FRACTION)
        sx = W / VIEWBOX_W

        size_vb = self.layout.text_size_vb if self.layout is not None else TEXT_SIZE_VB
        self._ensure_font(max(8, int(round(size_vb * sy))))
        left_path, right_path = self._screen_paths((sx, sy, band_top, self._font_size),
                                                   sx, sy, band_top)

        for is_left in (True, False):
            entries = [t for t in transcripts if (t[2] == "left") == is_left]
            if not entries:
                continue
            layer = np.zeros((H, W), dtype=np.uint8)
            drew = False
            for text, started_at, _side in entries:
                progress = (now - started_at) / SCROLL_DURATION
                if progress < 0.0 or progress > 1.0:
                    continue
                # left path scrolls outward-in, right path mirrors it
                offset = progress if is_left else (1.0 - progress)
                self._draw_text_on_path(layer, text,
                                        left_path if is_left else right_path, offset)
                drew = True
            if not drew:
                continue

            # Only touch the rows/columns the glyphs actually landed on —
            # the text band is a small slice of a full-screen surface.
            x, y, w, h = cv2.boundingRect(layer)
            if w == 0 or h == 0:
                continue
            pad = max(2, int(round(12 * sy)))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            sub = layer[y0:y1, x0:x1]

            alpha = sub.astype(np.float32) / 255.0
            alpha *= self._fade_mask(W, sx, 0.0, is_left)[None, x0:x1]
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
                shadow = np.maximum(shadow, cv2.GaussianBlur(shifted, (k, k), 0) * strength)
            shadow *= (1.0 - alpha)

            sh, sw = alpha.shape
            alpha_composite(surface_bgr, np.zeros((sh, sw, 3), dtype=np.uint8),
                            shadow, x0, y0)
            alpha_composite(surface_bgr, np.full((sh, sw, 3), 255, dtype=np.uint8),
                            alpha, x0, y0)
