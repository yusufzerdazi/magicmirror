"""Minimal SVG rasterizer for the mischief logo.

The web frontend shows `mischief.svg` through an <img> tag, so the local app
needs the same vector artwork rather than the thresholded JPEG fallback it used
before. Rather than pull in cairo/rsvg, this parses the single <path> in the
file and fills it with OpenCV — enough for this asset (moves, cubics, shorthand
cubics, horizontal lines, close) and nothing more.
"""

import re
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np

_NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_COMMAND = re.compile(r"[MmLlHhVvCcSsQqTtZz]")

# Curves are flattened into this many segments each; the logo is rendered small
# (a few percent of the screen) so this is far finer than one device pixel.
CURVE_SEGMENTS = 24

# Supersampling factor used when filling, then box-filtered down for antialiasing.
SUPERSAMPLE = 4


def _tokenize(d):
    """Split a path's `d` attribute into (command, [numbers]) pairs."""
    out = []
    pos = 0
    while True:
        m = _COMMAND.search(d, pos)
        if m is None:
            break
        nxt = _COMMAND.search(d, m.end())
        chunk = d[m.end():nxt.start() if nxt else len(d)]
        out.append((m.group(), [float(n.group()) for n in _NUMBER.finditer(chunk)]))
        pos = m.end()
        if nxt is None:
            break
    return out


def _cubic(p0, p1, p2, p3, segments=CURVE_SEGMENTS):
    """Flatten one cubic bezier into points, excluding the start point."""
    pts = []
    for i in range(1, segments + 1):
        t = i / segments
        u = 1.0 - t
        x = (u * u * u * p0[0] + 3 * u * u * t * p1[0]
             + 3 * u * t * t * p2[0] + t * t * t * p3[0])
        y = (u * u * u * p0[1] + 3 * u * u * t * p1[1]
             + 3 * u * t * t * p2[1] + t * t * t * p3[1])
        pts.append((x, y))
    return pts


def parse_path(d):
    """Return the path as a list of subpaths, each a list of (x, y) points."""
    subpaths = []
    current = []
    cursor = (0.0, 0.0)
    start = (0.0, 0.0)
    prev_ctrl = None  # second control point of the previous cubic, for S/s

    def flush():
        if len(current) >= 3:
            subpaths.append(list(current))

    for cmd, nums in _tokenize(d):
        rel = cmd.islower()
        upper = cmd.upper()

        if upper == "Z":
            flush()
            current = []
            cursor = start
            prev_ctrl = None
            continue

        if upper == "M":
            # A moveto ends the previous subpath; extra coordinate pairs after
            # the first are implicit linetos.
            for i in range(0, len(nums) - 1, 2):
                x, y = nums[i], nums[i + 1]
                if rel:
                    x, y = cursor[0] + x, cursor[1] + y
                if i == 0:
                    flush()
                    current = [(x, y)]
                    start = (x, y)
                else:
                    current.append((x, y))
                cursor = (x, y)
            prev_ctrl = None
            continue

        if upper in ("L", "T"):
            for i in range(0, len(nums) - 1, 2):
                x, y = nums[i], nums[i + 1]
                if rel:
                    x, y = cursor[0] + x, cursor[1] + y
                current.append((x, y))
                cursor = (x, y)
            prev_ctrl = None
            continue

        if upper in ("H", "V"):
            for n in nums:
                if upper == "H":
                    x = cursor[0] + n if rel else n
                    y = cursor[1]
                else:
                    x = cursor[0]
                    y = cursor[1] + n if rel else n
                current.append((x, y))
                cursor = (x, y)
            prev_ctrl = None
            continue

        if upper == "C":
            for i in range(0, len(nums) - 5, 6):
                c1 = (nums[i], nums[i + 1])
                c2 = (nums[i + 2], nums[i + 3])
                end = (nums[i + 4], nums[i + 5])
                if rel:
                    c1 = (cursor[0] + c1[0], cursor[1] + c1[1])
                    c2 = (cursor[0] + c2[0], cursor[1] + c2[1])
                    end = (cursor[0] + end[0], cursor[1] + end[1])
                current.extend(_cubic(cursor, c1, c2, end))
                cursor = end
                prev_ctrl = c2
            continue

        if upper in ("S", "Q"):
            # S/s: first control point is the reflection of the previous one.
            # Q/q is treated as a degenerate cubic (unused by this asset, but
            # cheap to support correctly).
            stride = 4
            for i in range(0, len(nums) - stride + 1, stride):
                a = (nums[i], nums[i + 1])
                end = (nums[i + 2], nums[i + 3])
                if rel:
                    a = (cursor[0] + a[0], cursor[1] + a[1])
                    end = (cursor[0] + end[0], cursor[1] + end[1])
                if upper == "S":
                    if prev_ctrl is None:
                        c1 = cursor
                    else:
                        c1 = (2 * cursor[0] - prev_ctrl[0], 2 * cursor[1] - prev_ctrl[1])
                    c2 = a
                else:
                    c1 = (cursor[0] + 2.0 / 3.0 * (a[0] - cursor[0]),
                          cursor[1] + 2.0 / 3.0 * (a[1] - cursor[1]))
                    c2 = (end[0] + 2.0 / 3.0 * (a[0] - end[0]),
                          end[1] + 2.0 / 3.0 * (a[1] - end[1]))
                current.extend(_cubic(cursor, c1, c2, end))
                cursor = end
                prev_ctrl = c2
            continue

    flush()
    return subpaths


def load_svg_paths(svg_path):
    """Return (subpaths, (viewbox_w, viewbox_h)) for an SVG of filled paths."""
    tree = ElementTree.parse(str(svg_path))
    root = tree.getroot()
    vb = root.get("viewBox")
    if vb:
        parts = [float(v) for v in re.split(r"[\s,]+", vb.strip())]
        vb_x, vb_y, vb_w, vb_h = parts
    else:
        vb_x, vb_y = 0.0, 0.0
        vb_w = float(root.get("width", 1))
        vb_h = float(root.get("height", 1))

    subpaths = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        d = el.get("d")
        if d:
            subpaths.extend(parse_path(d))

    if vb_x or vb_y:
        subpaths = [[(x - vb_x, y - vb_y) for x, y in sp] for sp in subpaths]
    return subpaths, (vb_w, vb_h)


def rasterize_alpha(svg_path, width, height):
    """Rasterize an SVG into a `height` x `width` uint8 alpha mask.

    The artwork is fitted with SVG's default `xMidYMid meet` behaviour — the
    same letterboxing a browser applies to `<img src="...svg">` — so the local
    coin matches the web one even though the logo is not square.
    """
    subpaths, (vb_w, vb_h) = load_svg_paths(svg_path)
    if not subpaths or width <= 0 or height <= 0:
        return np.zeros((max(height, 1), max(width, 1)), dtype=np.uint8)

    ss = SUPERSAMPLE
    big_w, big_h = width * ss, height * ss
    scale = min(big_w / vb_w, big_h / vb_h)
    off_x = (big_w - vb_w * scale) / 2.0
    off_y = (big_h - vb_h * scale) / 2.0

    canvas = np.zeros((big_h, big_w), dtype=np.uint8)
    contours = [
        np.round(np.array(sp, dtype=np.float64) * scale + (off_x, off_y)).astype(np.int32)
        for sp in subpaths
    ]
    # fillPoly over all contours at once gives even-odd behaviour, which cuts
    # the logo's interior shapes out as holes.
    cv2.fillPoly(canvas, contours, 255, lineType=cv2.LINE_8)
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)


def load_mischief_alpha(assets_dir, size):
    """Alpha mask for the mischief logo, preferring the SVG artwork."""
    assets_dir = Path(assets_dir)
    svg = assets_dir / "mischief.svg"
    if svg.exists():
        try:
            return rasterize_alpha(svg, size, size)
        except Exception as e:  # pragma: no cover - fall back rather than crash
            print(f"could not rasterize {svg.name} ({e}); falling back to raster asset",
                  flush=True)

    for name in ("mischief.png", "mischief.jpg"):
        path = assets_dir / name
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            _, alpha = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        return cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)

    return np.zeros((size, size), dtype=np.uint8)
