"""The DREAMWAVE stage in real 3D.

The generated picture is shown whole — a full square — standing between two
fluted pink marble pillars, under a dusk sky of volumetric clouds. On the left
a walnut cathedral radio floats on a cloud, its dial showing the station; on
the right, the poster's radio mast, with an eye on top that follows whoever is
in front of the mirror and sends out rings when someone is on air.

Everything here is rendered with OpenGL (moderngl, offscreen) on its own
thread, at about 30 frames a second: the display loop only copies the latest
finished frame and pastes the live picture into its window, so a busy GPU can
slow the clouds down but never the mirror.

The projection is head-coupled: the plane of the picture is fixed to the
screen, and the eye point moves a little with the person in front, so the
pillars, radio and clouds shift against one another as if the screen were a
window onto a real place.

World units: the screen is 1 unit tall, x to the right, y up, z towards the
viewer; the screen (and the picture) is the plane z = 0.
"""

import math
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import dreamwave as dw

EYE_Z = 1.75            # viewing distance, in screen heights
RENDER_FPS = 30.0
CLOUD_SCALE = 0.5       # clouds are raymarched at half resolution
MSAA = 4


# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------
class Layout:
    """Where the picture goes, in pixels and in world units.

    The picture fills the screen's height, centred and borderless; the pillars
    stand just outside it and the props use the bays beyond them."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.aspect = a = w / float(h)
        self.pic_px = dw.dream_stage_rect(w, h)
        sx, sy, s_px, _ = self.pic_px
        self.x0 = sx / h - a / 2
        self.x1 = (sx + s_px) / h - a / 2
        self.top = 0.5 - sy / h
        self.bottom = 0.5 - (sy + s_px) / h
        self.size = s_px / h
        self.gap = 0.0
        self.pillar_r = dw.PILLAR_R
        self.pillar_x = self.x1 + dw.PILLAR_GAP + self.pillar_r
        self.pillars = a - self.size >= 4 * self.pillar_r
        inner = self.pillar_x + self.pillar_r * 1.5
        self.bay = (inner, a / 2)
        self.bay_centre = (inner + a / 2) / 2
        self.bay_width = max(0.0, a / 2 - inner)
        self.props = self.bay_width >= 0.15

    def to_px(self, x, y):
        return (x + self.aspect / 2) * self.h, (0.5 - y) * self.h


# ----------------------------------------------------------------------------
# matrices
# ----------------------------------------------------------------------------
def off_axis(aspect, eye, near=0.05, far=80.0):
    """Projection * view for an eye looking through the screen rectangle."""
    ex, ey, ez = eye
    l = (-aspect / 2 - ex) * near / ez
    r = (aspect / 2 - ex) * near / ez
    b = (-0.5 - ey) * near / ez
    t = (0.5 - ey) * near / ez
    p = np.array([
        [2 * near / (r - l), 0, (r + l) / (r - l), 0],
        [0, 2 * near / (t - b), (t + b) / (t - b), 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]], np.float64)
    v = np.eye(4)
    v[:3, 3] = (-ex, -ey, -ez)
    return p @ v


def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def scale(x, y=None, z=None):
    return np.diag([x, x if y is None else y, x if z is None else z, 1.0])


def lin(rgb):
    """sRGB (0-1) to linear, for colours handed to the lighting."""
    return tuple(float(c) ** 2.2 for c in rgb)


def _set(prog, name, value):
    if name in prog:
        prog[name].value = value


def mat_bytes(m):
    return np.asarray(m, np.float32).T.tobytes()


# ----------------------------------------------------------------------------
# meshes (positions, normals, uvs, ao) -> interleaved float32 + uint32 indices
# ----------------------------------------------------------------------------
class Mesh:
    def __init__(self, pos, idx, uv=None, ao=None, normals=None):
        self.pos = np.asarray(pos, np.float32).reshape(-1, 3)
        self.idx = np.asarray(idx, np.uint32).reshape(-1)
        n = len(self.pos)
        self.uv = np.zeros((n, 2), np.float32) if uv is None else np.asarray(uv, np.float32)
        self.ao = np.ones((n, 1), np.float32) if ao is None else np.asarray(ao, np.float32).reshape(-1, 1)
        self.normals = self._smooth_normals() if normals is None else np.asarray(normals, np.float32)

    def _smooth_normals(self):
        tri = self.idx.reshape(-1, 3)
        a, b, c = self.pos[tri[:, 0]], self.pos[tri[:, 1]], self.pos[tri[:, 2]]
        fn = np.cross(b - a, c - a)
        n = np.zeros_like(self.pos)
        for k in range(3):
            np.add.at(n, tri[:, k], fn)
        length = np.linalg.norm(n, axis=1, keepdims=True)
        return (n / np.maximum(length, 1e-9)).astype(np.float32)

    def interleaved(self):
        return np.hstack([self.pos, self.normals, self.uv, self.ao]).astype(np.float32)

    @staticmethod
    def merge(meshes):
        pos, idx, uv, ao, nrm = [], [], [], [], []
        base = 0
        for m in meshes:
            pos.append(m.pos)
            idx.append(m.idx + base)
            uv.append(m.uv)
            ao.append(m.ao)
            nrm.append(m.normals)
            base += len(m.pos)
        return Mesh(np.vstack(pos), np.concatenate(idx), np.vstack(uv), np.vstack(ao),
                    np.vstack(nrm))


def grid_indices(rows, cols, wrap=True):
    """Quads over a rows x cols vertex grid (cols wraps around if wrap)."""
    idx = []
    cmax = cols if wrap else cols - 1
    for r in range(rows - 1):
        for c in range(cmax):
            c2 = (c + 1) % cols
            a, b = r * cols + c, r * cols + c2
            d, e = (r + 1) * cols + c, (r + 1) * cols + c2
            idx += [a, d, b, b, d, e]
    return idx


def lathe(profile, segments=96, radius_fn=None, ao_fn=None):
    """Surface of revolution of [(r, y), ...] (bottom to top) about the y axis."""
    rows = len(profile)
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    pos, ao, uv = [], [], []
    for i, (r, y) in enumerate(profile):
        rr = np.full(segments, r, np.float64)
        if radius_fn is not None:
            rr = radius_fn(i, y, th, rr)
        pos.append(np.stack([rr * np.sin(th), np.full(segments, y), rr * np.cos(th)], 1))
        ao.append(ao_fn(i, y, th) if ao_fn is not None else np.ones(segments))
        uv.append(np.stack([th / (2 * np.pi), np.full(segments, i / max(1, rows - 1))], 1))
    idx = np.asarray(grid_indices(rows, segments)).reshape(-1, 3)[:, ::-1]
    return Mesh(np.vstack(pos), idx, np.vstack(uv), np.concatenate(ao))


def box(x0, x1, y0, y1, z0, z1):
    """An axis-aligned box with flat normals."""
    faces = [
        ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (1, 0, 0)),
        ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (-1, 0, 0)),
        ((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0), (0, 1, 0)),
        ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0)),
        ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1)),
        ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (0, 0, -1)),
    ]
    pos, nrm, idx, uv = [], [], [], []
    for k, (a, b, c, d, n) in enumerate(faces):
        pos += [a, b, c, d]
        nrm += [n] * 4
        uv += [(0, 0), (1, 0), (1, 1), (0, 1)]
        idx += [4 * k, 4 * k + 1, 4 * k + 2, 4 * k, 4 * k + 2, 4 * k + 3]
    return Mesh(pos, idx, uv, normals=nrm)


def sphere(radius=1.0, rows=32, cols=48):
    pos, uv = [], []
    for i in range(rows + 1):
        v = i / rows
        phi = v * np.pi
        for j in range(cols):
            u = j / cols
            th = u * 2 * np.pi
            pos.append((radius * math.sin(phi) * math.sin(th), radius * math.cos(phi),
                        radius * math.sin(phi) * math.cos(th)))
            uv.append((u, v))
    pos = np.array(pos)
    return Mesh(pos, grid_indices(rows + 1, cols), uv,
                normals=(pos / max(radius, 1e-9)).astype(np.float32))


def cylinder_between(p0, p1, radius, sides=6):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    a = axis / length
    helper = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    pos, nrm = [], []
    for end in (p0, p1):
        for k in range(sides):
            t = 2 * np.pi * k / sides
            d = math.cos(t) * u + math.sin(t) * v
            pos.append(end + radius * d)
            nrm.append(d)
    return Mesh(pos, grid_indices(2, sides), normals=nrm)


def extrude_outline(outline, depth, bevel, rings=5):
    """A convex outline (CCW, in the xy plane) extruded along z with a rounded
    edge, front at z = +depth/2. UVs on the front face are the outline's
    bounding box, so a decal can be painted on it."""
    outline = np.asarray(outline, np.float64)
    n = len(outline)
    # Outward 2D normals at the vertices.
    prev = np.roll(outline, 1, 0)
    nxt = np.roll(outline, -1, 0)
    tangent = nxt - prev
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    out = np.stack([tangent[:, 1], -tangent[:, 0]], 1)
    lo, hi = outline.min(0), outline.max(0)

    pos, nrm, uv = [], [], []
    # Rings from the back face edge, round the side, to the front face edge.
    angles = list(np.linspace(-np.pi / 2, 0, rings)) + list(np.linspace(0, np.pi / 2, rings))
    half = depth / 2 - bevel
    for k, ang in enumerate(angles):
        inset = bevel * (1 - math.cos(ang))
        z = (half if ang >= 0 and k >= rings else -half) + bevel * math.sin(ang)
        ring = outline - out * inset
        for i in range(n):
            pos.append((ring[i, 0], ring[i, 1], z))
            nrm.append((out[i, 0] * math.cos(ang), out[i, 1] * math.cos(ang), math.sin(ang)))
            uv.append(((ring[i, 0] - lo[0]) / (hi[0] - lo[0]),
                       (ring[i, 1] - lo[1]) / (hi[1] - lo[1])))
    rows = len(angles)
    idx = grid_indices(rows, n)
    # Caps: fans from the centroid of the innermost rings.
    for face, ring_index, sign in ((1, rows - 1, 1), (-1, 0, -1)):
        start = ring_index * n
        ring = np.array(pos[start:start + n])
        centre = ring.mean(0)
        c_index = len(pos)
        pos.append(tuple(centre))
        nrm.append((0, 0, sign))
        uv.append(((centre[0] - lo[0]) / (hi[0] - lo[0]),
                   (centre[1] - lo[1]) / (hi[1] - lo[1])))
        cap_start = len(pos)
        for i in range(n):
            pos.append(tuple(ring[i]))
            nrm.append((0, 0, sign))
            uv.append(((ring[i, 0] - lo[0]) / (hi[0] - lo[0]),
                       (ring[i, 1] - lo[1]) / (hi[1] - lo[1])))
        for i in range(n):
            a, b = cap_start + i, cap_start + (i + 1) % n
            idx += [c_index, a, b] if face > 0 else [c_index, b, a]
    return Mesh(pos, idx, uv, normals=nrm)


def cathedral_outline(w, h, shoulder, points=120):
    """A cathedral radio's silhouette: straight sides up to the shoulder, then a
    gothic arch over the full width, CCW from the bottom right."""
    R = w * 0.9                          # each arc's radius
    cxr = w / 2 - R                      # right arc's centre (left of the edge)
    rise = math.sqrt(max(R * R - cxr * cxr, 1e-9))
    sy = (h - shoulder) / rise
    pts = [(w / 2, 0.0)]
    for k in range(1, 9):
        pts.append((w / 2, shoulder * k / 8))
    m = points // 2
    for k in range(1, m + 1):
        x = w / 2 - (w / 2) * k / m
        y = shoulder + math.sqrt(max(R * R - (x - cxr) ** 2, 0.0)) * sy
        pts.append((x, y))
    right_arc = pts[9:]
    pts += [(-x, y) for x, y in reversed(right_arc[:-1])]
    for k in range(8, -1, -1):
        pts.append((-w / 2, shoulder * k / 8))
    # The bottom edge, densified.
    for k in range(1, 12):
        pts.append((-w / 2 + w * k / 12, 0.0))
    out = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > 1e-7 or abs(p[1] - out[-1][1]) > 1e-7:
            out.append(p)
    if abs(out[0][0] - out[-1][0]) < 1e-7 and abs(out[0][1] - out[-1][1]) < 1e-7:
        out.pop()
    return out


# ----------------------------------------------------------------------------
# textures painted with PIL
# ----------------------------------------------------------------------------
def _fit(font_name, size):
    return dw.font(int(size), font_name)


def radio_front_textures(w_px=768, h_px=960, outline=None, body_w=1.0, body_h=1.0):
    """The radio's face: colour (RGBA over the walnut) and props
    (R metal, G cloth glow, B height)."""
    colour = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
    props = Image.new("RGB", (w_px, h_px), (0, 0, 0))
    dc, dp = ImageDraw.Draw(colour), ImageDraw.Draw(props)

    def to_px(x, y):    # body units, origin bottom centre
        return ((x / body_w + 0.5) * w_px, (1.0 - y / body_h) * h_px)

    # The grille: the outline inset, from just above the dial to near the top.
    outline = np.asarray(outline)
    grille = []
    for x, y in outline:
        gx, gy = x * 0.74, (y - body_h * 0.42) * 0.78 + body_h * 0.42
        if y >= body_h * 0.42:
            grille.append(to_px(gx, max(gy, body_h * 0.44)))
    grille = [p for p in grille]
    gold = (222, 178, 92, 255)
    cloth = (52, 30, 24, 255)
    dc.polygon(grille, fill=cloth)
    dp.polygon(grille, fill=(0, 255, 40))
    # Fretwork: a gothic window — mullions, arches and a rosette.
    xs = [p[0] for p in grille]
    ys = [p[1] for p in grille]
    gx0, gx1, gy0, gy1 = min(xs), max(xs), min(ys), max(ys)
    lw = max(3, w_px // 110)
    cx = (gx0 + gx1) / 2
    for k in range(1, 6):
        x = gx0 + (gx1 - gx0) * k / 6
        top = gy0 + (gy1 - gy0) * (0.42 if k in (1, 5) else 0.30 if k in (2, 4) else 0.22)
        dc.line([(x, top), (x, gy1)], fill=gold, width=lw)
        dp.line([(x, top), (x, gy1)], fill=(255, 0, 255), width=lw)
    for k in range(3):
        rr = (gx1 - gx0) * (0.17 + 0.09 * k)
        box_ = (cx - rr, gy0 + (gy1 - gy0) * 0.36 - rr, cx + rr, gy0 + (gy1 - gy0) * 0.36 + rr)
        dc.arc(box_, 180, 360, fill=gold, width=lw)
        dp.arc(box_, 180, 360, fill=(255, 0, 255), width=lw)
    rc = (cx, gy0 + (gy1 - gy0) * 0.25)
    rr = (gx1 - gx0) * 0.09
    dc.ellipse((rc[0] - rr, rc[1] - rr, rc[0] + rr, rc[1] + rr), outline=gold, width=lw)
    dp.ellipse((rc[0] - rr, rc[1] - rr, rc[0] + rr, rc[1] + rr), outline=(255, 0, 255), width=lw)
    for a in range(0, 360, 45):
        t = math.radians(a)
        dc.line([rc, (rc[0] + rr * math.cos(t), rc[1] + rr * math.sin(t))], fill=gold, width=lw - 1)
        dp.line([rc, (rc[0] + rr * math.cos(t), rc[1] + rr * math.sin(t))], fill=(255, 0, 255), width=lw - 1)
    # Gold trim around the grille.
    dc.line(grille + [grille[0]], fill=gold, width=lw * 2)
    dp.line(grille + [grille[0]], fill=(255, 0, 255), width=lw * 2)
    # Escutcheons round the knobs.
    for kx in (-0.30, 0.0, 0.30):
        x, y = to_px(kx * body_w, body_h * 0.13)
        r = w_px * (0.075 if kx == 0 else 0.06)
        dc.ellipse((x - r, y - r, x + r, y + r), outline=gold, width=lw)
        dp.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 255), width=lw)
    # The maker's plate.
    f = _fit(dw.DISPLAY_FONT, w_px * 0.065)
    text = "DREAMWAVE"
    tw = dc.textlength(text, font=f)
    x, y = to_px(0, body_h * 0.405)
    dc.text((x - tw / 2, y - w_px * 0.05), text, font=f, fill=gold)
    dp.text((x - tw / 2, y - w_px * 0.05), text, font=f, fill=(255, 0, 200))
    props = props.filter(ImageFilter.GaussianBlur(1.2))
    return colour, props


def dial_texture(freq, on_air, size=(512, 256)):
    """The illuminated dial: a half-moon of amber glass with the FM scale."""
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Glow.
    glow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(glow).ellipse((w * 0.08, h * 0.18, w * 0.92, h * 1.9), fill=255)
    glow = glow.filter(ImageFilter.GaussianBlur(h * 0.18))
    base = Image.new("RGBA", (w, h), (255, 196, 110, 255))
    dark = Image.new("RGBA", (w, h), (120, 52, 30, 255))
    img = Image.composite(base, dark, glow)
    d = ImageDraw.Draw(img)
    cx, cy = w / 2, h * 0.98
    r_outer, r_inner = h * 0.86, h * 0.62
    ink = (60, 26, 16, 255)
    small = dw.font(int(h * 0.12), dw.DIAL_FONT)
    for f10 in range(880, 1081, 5):
        f = f10 / 10.0
        t = (f - dw.DIAL_MIN) / (dw.DIAL_MAX - dw.DIAL_MIN)
        ang = math.pi * (1.0 - (0.08 + 0.84 * t))
        major = f10 % 40 == 0
        r0 = r_outer - (h * 0.12 if major else h * 0.06)
        d.line([(cx + r0 * math.cos(ang), cy - r0 * math.sin(ang)),
                (cx + r_outer * math.cos(ang), cy - r_outer * math.sin(ang))],
               fill=ink, width=3 if major else 2)
        if major:
            label = str(int(f))
            tw = d.textlength(label, font=small)
            rl = r_inner - h * 0.04
            d.text((cx + rl * math.cos(ang) - tw / 2, cy - rl * math.sin(ang) - h * 0.07),
                   label, font=small, fill=ink)
    # The station's frequency, big, under the arc.
    big = dw.font(int(h * 0.26), dw.DIAL_FONT)
    label = f"{freq:5.1f}"
    tw = d.textlength(label, font=big)
    d.text((cx - tw / 2, h * 0.58), label, font=big, fill=(70, 20, 12, 255))
    # Needle.
    t = (freq - dw.DIAL_MIN) / (dw.DIAL_MAX - dw.DIAL_MIN)
    ang = math.pi * (1.0 - (0.08 + 0.84 * min(max(t, 0), 1)))
    d.line([(cx, cy), (cx + r_outer * 1.02 * math.cos(ang), cy - r_outer * 1.02 * math.sin(ang))],
           fill=(200, 30, 30, 255), width=5)
    d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=(60, 20, 12, 255))
    if on_air:
        f = dw.font(int(h * 0.11), dw.DISPLAY_FONT)
        tw = d.textlength("ON AIR", font=f)
        d.rounded_rectangle((cx - tw / 2 - 8, h * 0.06, cx + tw / 2 + 8, h * 0.06 + h * 0.15),
                            radius=6, fill=(214, 40, 40, 255))
        d.text((cx - tw / 2, h * 0.065), "ON AIR", font=f, fill=(255, 238, 210, 255))
    # Half-moon mask.
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((cx - w * 0.5, cy - h * 0.98, cx + w * 0.5, cy + h * 0.98), fill=255)
    img.putalpha(mask)
    return img


def _wrap_to(d, text, width, font_name, size, min_size, max_lines=2):
    """The biggest size at which text fits in max_lines lines of width."""
    while size >= min_size:
        f = dw.font(size, font_name)
        words, lines, line = text.split(), [], ""
        for word in words:
            trial = (line + " " + word).strip()
            if d.textlength(trial, font=f) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        if len(lines) <= max_lines and all(d.textlength(l, font=f) <= width for l in lines):
            return f, lines
        size -= 2
    f = dw.font(min_size, font_name)
    lines = lines[:max_lines]
    while lines and d.textlength(lines[-1] + "\u2026", font=f) > width and len(lines[-1]) > 1:
        lines[-1] = lines[-1][:-1]
    lines[-1] = lines[-1].rstrip() + "\u2026"
    return f, lines


def caption_image(width_px, title, subtitle, on_air, height_px=None):
    """The station's name (or what was said) above the radio."""
    width_px = int(width_px)
    h = height_px or int(width_px * 0.52)
    img = Image.new("RGBA", (width_px, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    small = dw.font(max(10, int(width_px * 0.075)), dw.DISPLAY_FONT)
    head = "\u25cf ON AIR" if on_air else "DREAMWAVE FM"
    colour = (246, 96, 86, 255) if on_air else dw.GOLD + (255,)
    tw = d.textlength(head, font=small)
    d.text(((width_px - tw) / 2, 0), head, font=small, fill=colour)
    top = int(width_px * 0.11)
    big, lines = _wrap_to(d, title, width_px * 0.96, dw.DISPLAY_FONT,
                          int(width_px * 0.15), max(10, int(width_px * 0.075)))
    line_h = int(big.size * 1.12)
    y = top + (h - top - int(width_px * 0.11) - line_h * len(lines)) // 2
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    for k, line in enumerate(lines):
        tw = d.textlength(line, font=big)
        sd.text(((width_px - tw) / 2 + 2, y + k * line_h + 3), line, font=big,
                fill=(20, 8, 30, 170))
    img = Image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(3)), img)
    d = ImageDraw.Draw(img)
    for k, line in enumerate(lines):
        tw = d.textlength(line, font=big)
        d.text(((width_px - tw) / 2, y + k * line_h), line, font=big, fill=dw.CREAM + (255,))
    if subtitle:
        f = dw.font(max(10, int(width_px * 0.07)), dw.DIAL_FONT)
        tw = d.textlength(subtitle, font=f)
        d.text(((width_px - tw) / 2 + 1, h - int(width_px * 0.09) + 2), subtitle, font=f,
               fill=(20, 8, 30, 170))
        d.text(((width_px - tw) / 2, h - int(width_px * 0.09)), subtitle, font=f,
               fill=dw.CREAM + (235,))
    return img


def constellation_texture(w, h):
    """Gold constellations and astrolabe rings, as on the poster."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(3)
    unit = h
    for (ax, ay) in ((0.08 * w, 0.18 * h), (0.92 * w, 0.22 * h)):
        for k, rad in enumerate((0.12, 0.17, 0.23, 0.27)):
            rr = rad * unit
            d.ellipse((ax - rr, ay - rr, ax + rr, ay + rr),
                      outline=dw.GOLD + (60 if k % 2 else 40,), width=1)
        for a in range(0, 360, 10):
            t = math.radians(a)
            d.line((ax + math.cos(t) * 0.23 * unit, ay + math.sin(t) * 0.23 * unit,
                    ax + math.cos(t) * 0.27 * unit, ay + math.sin(t) * 0.27 * unit),
                   fill=dw.GOLD + (50,), width=1)
    for _ in range(9):
        x, y = rng.uniform(0, w), rng.uniform(0, h * 0.55)
        pts = [(x, y)]
        for _ in range(rng.integers(3, 6)):
            ang = rng.uniform(0, 2 * np.pi)
            dd = rng.uniform(0.03, 0.07) * unit
            pts.append((pts[-1][0] + math.cos(ang) * dd, pts[-1][1] + math.sin(ang) * dd))
        d.line(pts, fill=dw.GOLD + (90,), width=1)
        for qx, qy in pts:
            r = rng.uniform(1.8, 3.2)
            d.ellipse((qx - r, qy - r, qx + r, qy + r), fill=dw.GOLD + (220,))
    return img


def noise_volume(n=64, seed=7):
    """Tileable 3D noise: R low, G mid, B high frequency, A billowy."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.fftfreq(n)
    fx, fy, fz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    f2 = fx * fx + fy * fy + fz * fz
    out = []
    for cutoff in (0.045, 0.09, 0.18):
        white = rng.standard_normal((n, n, n))
        spec = np.fft.fftn(white) * np.exp(-f2 / (2 * cutoff * cutoff))
        field = np.real(np.fft.ifftn(spec))
        field = (field - field.mean()) / (field.std() * 3.2) + 0.5
        out.append(np.clip(field, 0, 1))
    billow = 1.0 - np.abs(out[1] * 2 - 1)
    out.append(np.clip(billow, 0, 1))
    vol = np.stack(out, -1)
    return (vol * 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# shaders
# ----------------------------------------------------------------------------
COMMON = """
vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
uniform float exposure;
vec3 finish(vec3 hdr) { return pow(aces(hdr * exposure), vec3(1.0 / 2.2)); }
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
// The dusk sky, by direction: used for the backdrop and for reflections.
vec3 lin(vec3 c) { return pow(c, vec3(2.2)); }
vec3 sky_colour(vec3 d) {
    float v = clamp(d.y * 2.2 + 0.45, 0.0, 1.0);
    vec3 low = lin(vec3(0.93, 0.50, 0.46));
    vec3 mid = lin(vec3(0.42, 0.21, 0.42));
    vec3 high = lin(vec3(0.09, 0.07, 0.19));
    vec3 c = mix(low, mid, smoothstep(0.0, 0.5, v));
    c = mix(c, high, smoothstep(0.45, 1.0, v));
    return c;
}
"""

SKY_VS = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() { v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

SKY_FS = """
#version 330
in vec2 v_uv;
out vec4 frag;
uniform vec3 eye;
uniform float aspect;
uniform float time;
uniform vec2 screen;
uniform vec2 sun_px;
uniform sampler2D clouds;
uniform sampler2D stars_tex;
""" + COMMON + """
void main() {
    vec3 p = vec3((v_uv.x - 0.5) * aspect, v_uv.y - 0.5, 0.0);
    vec3 d = normalize(p - eye);
    vec3 c = sky_colour(d);
    // Warm light from behind the picture, as if the portal were lit.
    vec2 px = v_uv * screen;
    float r = length((px - sun_px) / screen.y);
    c += lin(vec3(1.0, 0.62, 0.52)) * exp(-r * r / 0.10) * 0.45;
    c += lin(vec3(1.0, 0.80, 0.62)) * exp(-r * r / 0.012) * 0.2;
    // Stars, fixed to the sky (they shift with the eye as if infinitely far).
    vec2 sp = vec2(atan(d.x, -d.z), asin(clamp(d.y, -1.0, 1.0))) * 260.0;
    vec2 cell = floor(sp);
    float h = hash12(cell);
    vec2 off = vec2(hash12(cell + 7.1), hash12(cell + 3.7)) - 0.5;
    float dist = length(fract(sp) - 0.5 - off * 0.6);
    float star = step(0.986, h) * smoothstep(0.16, 0.0, dist);
    float tw = 0.6 + 0.4 * sin(time * (1.5 + h * 4.0) + h * 40.0);
    c += vec3(1.0, 0.92, 0.85) * star * tw * smoothstep(-0.05, 0.2, d.y + 0.15) * 1.4;
    // Constellations and rings, far away.
    vec2 cuv = v_uv + eye.xy * vec2(-0.02 / aspect, -0.02);
    vec4 cs = texture(stars_tex, vec2(cuv.x, 1.0 - cuv.y));
    c = mix(c, lin(vec3(0.95, 0.76, 0.42)), cs.a * 0.85);
    // Halftone dots towards the horizon.
    float pitch = screen.y / 130.0;
    vec2 g = px / pitch;
    vec2 gc = vec2(fract(g.x + 0.5 * mod(floor(g.y), 2.0)), fract(g.y)) - 0.5;
    float k = smoothstep(0.30, 0.0, v_uv.y);
    float dot_ = smoothstep(0.05, -0.05, length(gc) - 0.48 * k);
    c = mix(c, lin(vec3(0.98, 0.58, 0.50)), dot_ * 0.22);
    // Clouds, premultiplied, raymarched at half resolution.
    vec4 cl = texture(clouds, v_uv);
    c = cl.rgb + c * (1.0 - cl.a);
    // Paper grain.
    c *= 1.0 + (hash12(px + fract(time) * 100.0) - 0.5) * 0.035;
    frag = vec4(finish(c), 1.0);
}
"""

CLOUD_VS = """
#version 330
in vec2 in_pos;
uniform mat4 viewproj;
uniform vec4 cloud;      // centre xyz, radius
out vec3 v_world;
void main() {
    // A quad in front of the bounding sphere, big enough for oblique views.
    vec3 w = cloud.xyz + vec3(in_pos * cloud.w * 2.1, cloud.w);
    v_world = w;
    gl_Position = viewproj * vec4(w, 1.0);
}
"""

CLOUD_FS = """
#version 330
in vec3 v_world;
out vec4 frag;
uniform vec3 eye;
uniform vec4 cloud;
uniform vec4 puffs[16];
uniform int puff_count;
uniform float base_y;
uniform float seed;
uniform float time;
uniform float density;
uniform sampler3D noise;
""" + COMMON + """
float shape(vec3 q) {
    float f = 0.0;
    for (int i = 0; i < 16; i++) {
        if (i >= puff_count) break;
        float d = length(q - puffs[i].xyz) / puffs[i].w;
        f = max(f, 1.0 - d);
    }
    return f;
}
float dens(vec3 q) {
    float f = shape(q);
    if (f <= -0.12) return 0.0;
    // Wisps only at the edges: the bulbs themselves stay round and solid.
    vec4 n1 = texture(noise, q * 0.55 + vec3(seed, seed * 0.7, time * 0.004));
    vec4 n2 = texture(noise, q * 1.05 + vec3(time * 0.008, seed, 0.0));
    float n = n1.a * 0.65 + n2.b * 0.35;
    float edge = 1.0 - smoothstep(0.0, 0.2, f);
    float d = smoothstep(0.0, 0.16, f + (n - 0.55) * (0.03 + 0.16 * edge));
    float base = base_y + (n1.r - 0.5) * 0.16;
    d *= smoothstep(base - 0.06, base + 0.14, q.y);
    return d;
}
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
// Shadows come from the smooth shape alone: bulbs shade as bulbs.
float dens_light(vec3 q) {
    return smoothstep(-0.02, 0.18, shape(q)) * smoothstep(base_y - 0.06, base_y + 0.14, q.y);
}
void main() {
    vec3 ro = eye;
    vec3 rd = normalize(v_world - eye);
    vec3 oc = ro - cloud.xyz;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - cloud.w * cloud.w;
    float disc = b * b - c;
    if (disc <= 0.0) discard;
    float s = sqrt(disc);
    float t0 = max(-b - s, 0.0), t1 = -b + s;
    const int STEPS = 44;
    float dt = (t1 - t0) / float(STEPS);
    float t = t0 + dt * ign(gl_FragCoord.xy);
    vec3 L = normalize(vec3(-0.78, 0.52, 0.18));      // the sun, low on the left
    vec3 sun = lin(vec3(1.0, 0.88, 0.80)) * 1.5;
    vec3 U = normalize(vec3(0.25, -0.9, 0.35));       // sunset glow from below
    vec3 under = lin(vec3(1.0, 0.50, 0.58)) * 0.75;
    vec3 sky_top = lin(vec3(0.58, 0.52, 0.80)) * 0.32;
    vec3 sky_low = lin(vec3(0.60, 0.32, 0.46)) * 0.22;
    float mu = dot(rd, L);
    float g = 0.3;
    float phase = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5) * 0.3 + 0.55;
    vec3 col = vec3(0.0);
    float T = 1.0;
    float inv_r = 1.0 / cloud.w;
    for (int i = 0; i < STEPS; i++) {
        vec3 q = (ro + rd * t - cloud.xyz) * inv_r;
        float d = dens(q);
        if (d > 0.003) {
            float tau_l = dens_light(q + L * 0.05) + dens_light(q + L * 0.12)
                        + dens_light(q + L * 0.24) + dens_light(q + L * 0.42) * 0.8;
            float tau_u = dens_light(q + U * 0.10) + dens_light(q + U * 0.30);
            // Light that got in, with the usual cheap multiple-scattering
            // octaves so the shadowed side is soft rather than black.
            float tl = tau_l * density * 0.05;
            float tu = tau_u * density * 0.05;
            float beer = exp(-tl) + 0.3 * exp(-tl * 0.25);
            float beer_u = exp(-tu) + 0.3 * exp(-tu * 0.3);
            float powder = 1.0 - exp(-d * 6.0);
            float hgt = clamp(q.y * 0.9 + 0.45, 0.0, 1.0);
            vec3 amb = mix(sky_low, sky_top, hgt);
            vec3 light = sun * beer * phase * mix(1.0, powder, 0.6)
                       + under * beer_u * (0.6 + 0.4 * powder) + amb;
            float tau = d * density * dt * inv_r;
            float a = 1.0 - exp(-tau);
            col += T * a * light;
            T *= 1.0 - a;
            if (T < 0.02) break;
        }
        t += dt;
    }
    if (T > 0.995) discard;
    frag = vec4(col, 1.0 - T);
}
"""

MESH_VS = """
#version 330
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
in float in_ao;
uniform mat4 viewproj;
uniform mat4 model;
out vec3 v_world;
out vec3 v_normal;
out vec3 v_local;
out vec2 v_uv;
out float v_ao;
void main() {
    vec4 w = model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_normal = mat3(model) * in_normal;
    v_local = in_pos;
    v_uv = in_uv;
    v_ao = in_ao;
    gl_Position = viewproj * w;
}
"""

MESH_FS = """
#version 330
in vec3 v_world;
in vec3 v_normal;
in vec3 v_local;
in vec2 v_uv;
in float v_ao;
out vec4 frag;
uniform vec3 eye;
uniform int material;      // 0 marble, 1 gold, 2 walnut, 3 radio face, 4 emissive tex,
                           // 5 eye, 6 bakelite, 7 glow orb, 8 brass lamp
uniform vec3 tint;
uniform float glow;
uniform float time;
uniform sampler3D noise;
uniform sampler2D tex_a;
uniform sampler2D tex_b;
uniform vec3 gaze;         // the eye's forward axis, in its local space
// tint is linear
""" + COMMON + """
const float PI = 3.14159265;
float D_ggx(float nh, float a) { float a2 = a * a; float d = nh * nh * (a2 - 1.0) + 1.0; return a2 / (PI * d * d); }
vec3 F_schlick(float vh, vec3 f0) { return f0 + (1.0 - f0) * pow(1.0 - vh, 5.0); }
vec3 light_it(vec3 albedo, vec3 N, vec3 V, float rough, float metal, float ao) {
    vec3 f0 = mix(vec3(0.04), albedo, metal);
    vec3 diff = albedo * (1.0 - metal);
    vec3 Ls[3] = vec3[3](normalize(vec3(-0.55, 0.55, 0.75)),
                         normalize(vec3(0.75, 0.1, -0.65)),
                         normalize(vec3(0.2, -0.6, 0.5)));
    vec3 Cs[3] = vec3[3](lin(vec3(1.0, 0.92, 0.86)) * 3.0,
                         lin(vec3(1.0, 0.62, 0.52)) * 2.6,
                         lin(vec3(0.95, 0.55, 0.60)) * 0.7);
    vec3 c = vec3(0.0);
    float a = max(rough * rough, 0.02);
    for (int i = 0; i < 3; i++) {
        vec3 L = Ls[i];
        float nl = max(dot(N, L), 0.0);
        if (nl <= 0.0) continue;
        vec3 H = normalize(L + V);
        float nh = max(dot(N, H), 0.0), vh = max(dot(V, H), 0.0);
        vec3 F = F_schlick(vh, f0);
        float spec = D_ggx(nh, a) * 0.25 / max(nl * max(dot(N, V), 0.05), 0.05) * nl;
        c += Cs[i] * (diff / PI * nl * (1.0 - F) + F * spec * nl);
    }
    vec3 up = lin(vec3(0.50, 0.42, 0.66)), down = lin(vec3(0.62, 0.40, 0.40));
    vec3 amb = mix(down, up, N.y * 0.5 + 0.5) * 0.7;
    vec3 R = reflect(-V, N);
    // Reflections: the dusk sky, warmed by the lit portal behind and the room.
    vec3 env = sky_colour(R) * 1.2 + lin(vec3(1.0, 0.78, 0.55)) * 1.4 * pow(max(R.y * 0.5 + 0.55, 0.0), 3.0)
             + lin(vec3(1.0, 0.70, 0.55)) * 0.6;
    vec3 Fe = F_schlick(max(dot(N, V), 0.0), f0) * mix(1.0, 0.35, rough);
    c += (diff * amb + env * Fe) * ao;
    return c;
}
float fbm(vec3 p) {
    return texture(noise, p).r * 0.5 + texture(noise, p * 2.03).g * 0.3 + texture(noise, p * 4.1).b * 0.2;
}
void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(eye - v_world);
    vec3 col;
    if (material == 0) {
        // Pink marble with pale veins.
        vec3 p = v_local * 2.2;
        float n = fbm(p * 0.6);
        float vein = abs(sin((p.y * 1.3 + p.x * 2.0 + n * 9.0)));
        vein = smoothstep(0.92, 1.0, 1.0 - vein) * 0.65 + smoothstep(0.6, 1.0, fbm(p * 1.7)) * 0.2;
        vec3 albedo = mix(tint, lin(vec3(0.98, 0.90, 0.86)), vein);
        albedo *= 0.92 + 0.16 * n;
        col = light_it(albedo, N, V, 0.32, 0.0, v_ao);
    } else if (material == 1) {
        col = light_it(tint, N, V, 0.28, 1.0, v_ao);
        col += tint * glow * 2.0;
    } else if (material == 2 || material == 3) {
        // Walnut: long grain along y, figured with noise.
        vec3 p = v_local * vec3(9.0, 1.6, 9.0);
        float n = fbm(p * 0.5);
        float streak = texture(noise, v_local * vec3(3.0, 0.25, 3.0) * 4.0).b;
        float grain = smoothstep(0.35, 0.75, streak) * 0.6 + n * 0.4;
        vec3 albedo = mix(lin(vec3(0.30, 0.15, 0.08)), lin(vec3(0.55, 0.30, 0.15)), grain);
        float rough = 0.3, metal = 0.0;
        vec3 emit = vec3(0.0);
        if (material == 3) {
            vec4 dc = texture(tex_a, vec2(v_uv.x, 1.0 - v_uv.y));
            vec3 pr = texture(tex_b, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;
            albedo = mix(albedo, lin(dc.rgb), dc.a);
            metal = pr.r;
            rough = mix(0.3, 0.32, pr.r);
            // Bump from the height channel.
            vec2 e = vec2(1.0 / 768.0, 1.0 / 960.0) * 1.5;
            float hx = texture(tex_b, vec2(v_uv.x + e.x, 1.0 - v_uv.y)).b - texture(tex_b, vec2(v_uv.x - e.x, 1.0 - v_uv.y)).b;
            float hy = texture(tex_b, vec2(v_uv.x, 1.0 - v_uv.y - e.y)).b - texture(tex_b, vec2(v_uv.x, 1.0 - v_uv.y + e.y)).b;
            N = normalize(N + vec3(-hx, -hy, 0.0) * 1.6);
            // The cloth glows from inside when the host is talking.
            float flick = 0.85 + 0.15 * sin(time * 23.0) * sin(time * 7.3);
            emit = vec3(1.0, 0.36, 0.08) * pr.g * glow * flick * 1.2 * (1.0 - pr.r);
            emit += vec3(1.0, 0.30, 0.06) * pr.g * 0.05;
        }
        col = light_it(albedo, N, V, rough, metal, v_ao) + emit;
    } else if (material == 4) {
        vec4 t = texture(tex_a, vec2(v_uv.x, 1.0 - v_uv.y));
        if (t.a < 0.5) discard;
        col = lin(t.rgb) * (1.3 + glow * 0.7);
        // Glass: a soft specular sheen.
        vec3 H = normalize(normalize(vec3(-0.5, 0.6, 0.7)) + V);
        col += vec3(1.0) * pow(max(dot(N, H), 0.0), 60.0) * 0.6;
    } else if (material == 5) {
        // The eye: sclera, iris and pupil around the gaze axis.
        vec3 n = normalize(v_local);
        float ang = acos(clamp(dot(n, normalize(gaze)), -1.0, 1.0));
        vec3 sclera = vec3(0.95, 0.90, 0.84);
        float vein = smoothstep(0.75, 1.0, fbm(n * 3.0)) * smoothstep(0.9, 1.6, ang);
        sclera = mix(sclera, vec3(0.85, 0.35, 0.35), vein * 0.5);
        vec3 side = normalize(cross(normalize(gaze), vec3(0.0, 1.0, 0.0)) + 1e-4);
        float around = atan(dot(n, side), dot(n, cross(normalize(gaze), side)));
        float stripes = 0.5 + 0.5 * sin(around * 38.0 + fbm(n * 6.0) * 6.0);
        vec3 iris = mix(vec3(0.08, 0.35, 0.40), vec3(0.85, 0.62, 0.22), smoothstep(0.30, 0.10, ang) * 0.7 + stripes * 0.25);
        vec3 albedo = sclera;
        albedo = mix(albedo, iris, smoothstep(0.47, 0.44, ang));
        albedo = mix(albedo, vec3(0.05, 0.05, 0.06), smoothstep(0.50, 0.46, ang) * smoothstep(0.40, 0.46, ang) * 0.8);
        albedo = mix(albedo, vec3(0.01), smoothstep(0.19, 0.16, ang));
        col = light_it(albedo, N, V, 0.12, 0.0, 1.0);
        col += vec3(1.0, 0.7, 0.4) * glow * smoothstep(0.47, 0.2, ang) * 0.5;
    } else if (material == 6) {
        // Dark bakelite with knurling catching the light.
        col = light_it(tint, N, V, 0.22, 0.0, v_ao);
    } else if (material == 7) {
        // Glowing orb: gold glass lit from within.
        float fres = pow(1.0 - max(dot(N, V), 0.0), 2.0);
        col = light_it(tint * 0.6, N, V, 0.15, 0.6, 1.0);
        col += tint * (0.25 + glow * 1.6) * (1.0 - fres * 0.6);
    } else {
        col = light_it(tint, N, V, 0.4, 0.0, v_ao);
    }
    frag = vec4(finish(col), 1.0);
}
"""

RING_VS = """
#version 330
in vec2 in_pos;
uniform mat4 viewproj;
uniform vec4 ring;      // centre xyz, radius
out vec2 v_p;
void main() {
    v_p = in_pos;
    vec3 w = ring.xyz + vec3(in_pos * ring.w, 0.0);
    gl_Position = viewproj * vec4(w, 1.0);
}
"""

RING_FS = """
#version 330
in vec2 v_p;
out vec4 frag;
uniform float alpha;
uniform vec3 colour;
uniform float soft;
void main() {
    float r = length(v_p);
    float ring = exp(-pow((r - 0.9) / soft, 2.0));
    float halo = exp(-r * r * 3.0) * 0.0;
    frag = vec4(colour * (ring + halo) * alpha, 0.0);
}
"""

PRESENT_FS = """
#version 330
in vec2 v_uv;
out vec4 frag;
uniform sampler2D scene;
void main() {
    // Flip for top-down rows and swap to BGR for OpenCV.
    vec3 c = texture(scene, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;
    frag = vec4(c.bgr, 1.0);
}
"""

SPRITE_VS = """
#version 330
in vec2 in_pos;
uniform vec4 rect;      // x, y, w, h in output pixels (top-left origin)
uniform vec2 screen;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    vec2 px = rect.xy + vec2(v_uv.x, v_uv.y) * rect.zw;
    // Output rows are top-down after the flip in PRESENT, so y maps directly.
    vec2 ndc = vec2(px.x / screen.x * 2.0 - 1.0, px.y / screen.y * 2.0 - 1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

SPRITE_FS = """
#version 330
in vec2 v_uv;
out vec4 frag;
uniform sampler2D image;
uniform float opacity;
void main() {
    vec4 t = texture(image, v_uv);
    frag = vec4(t.bgr, t.a * opacity);
}
"""


# ----------------------------------------------------------------------------
# clouds
# ----------------------------------------------------------------------------
class Cloud:
    def __init__(self, rng, layout, eye_z, near=False, x=None):
        self.rng = rng
        a = layout.aspect
        if near:
            self.z = rng.uniform(-0.6, -0.25)
        else:
            self.z = rng.uniform(-4.0, -0.8)
        k = (eye_z - self.z) / eye_z        # how much bigger the view is at this depth
        # Size chosen by how big it looks: a tenth to a fifth of the screen.
        self.radius = rng.uniform(0.10, 0.2) * k
        span = a / 2 * k + self.radius
        self.span = span
        self.x = rng.uniform(-span, span) if x is None else x
        self.y = rng.uniform(-0.45, 0.42) * k if not near else 0.0
        self.speed = rng.uniform(0.006, 0.016) * (1 if rng.random() < 0.5 else -1) * k ** 0.5
        if near:
            self.speed = 0.0
        self.seed = rng.uniform(0, 10)
        self.base = -0.30
        puffs = []
        n_base = int(rng.integers(4, 6))
        for i in range(n_base):
            t = (i + 0.5) / n_base * 2 - 1
            r = rng.uniform(0.26, 0.34) * (1.1 - 0.35 * abs(t))
            puffs.append((t * 0.58 + rng.uniform(-0.05, 0.05), self.base + r * 0.85,
                          rng.uniform(-0.12, 0.12), r))
        for _ in range(int(rng.integers(5, 7))):
            px = rng.uniform(-0.42, 0.42)
            r = rng.uniform(0.17, 0.25) * (1.1 - 0.6 * abs(px))
            py = self.base + 0.42 + (0.22 - abs(px) * 0.35) + rng.uniform(-0.04, 0.06)
            puffs.append((px, py, rng.uniform(-0.12, 0.12), r))
        while len(puffs) < 16:
            px = rng.uniform(-0.3, 0.3)
            r = rng.uniform(0.09, 0.14)
            py = self.base + 0.62 + (0.2 - abs(px) * 0.5) + rng.uniform(-0.03, 0.05)
            puffs.append((px, py, rng.uniform(-0.15, 0.15), r))
        self.puffs = np.array(puffs[:16], np.float32)
        self.density = rng.uniform(30, 40)

    def update(self, dt):
        self.x += self.speed * dt
        if self.x > self.span:
            self.x = -self.span
        elif self.x < -self.span:
            self.x = self.span


# ----------------------------------------------------------------------------
# the renderer
# ----------------------------------------------------------------------------
class Stage3D:
    """Runs the 3D stage on its own thread; `latest()` is the newest frame."""

    def __init__(self, fps=RENDER_FPS):
        self.fps = fps
        self.lock = threading.Lock()
        self.state = {
            "freq": 88.1, "name": "", "on_air": False, "heard": "",
            "signal": 0.0, "static": 0.0, "speaking": False, "viewer": 0.0,
            "viewer_y": 0.0, "present": False,
        }
        self.size = None
        self._frame = None
        self._frame_size = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.failed = None
        self.render_ms = 0.0
        self._thread = None

    # -- main thread -------------------------------------------------------
    def start(self, w, h):
        self.size = (w, h)
        self._thread = threading.Thread(target=self._run, name="stage3d", daemon=True)
        self._thread.start()
        return True

    def resize(self, w, h):
        with self.lock:
            self.size = (w, h)

    def update(self, **state):
        with self.lock:
            self.state.update(state)

    def latest(self):
        with self.lock:
            return self._frame

    def layout(self, w, h):
        return Layout(w, h)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- render thread -----------------------------------------------------
    def _run(self):
        try:
            import moderngl
            self.ctx = moderngl.create_standalone_context(require=330)
            self._build()
        except Exception as e:
            self.failed = f"{type(e).__name__}: {e}"
            print(f"stage3d: unavailable ({self.failed}); using the flat stage", flush=True)
            return
        print(f"stage3d: {self.ctx.info['GL_RENDERER']}", flush=True)
        period = 1.0 / self.fps
        last = time.time()
        while not self._stop.is_set():
            start = time.time()
            dt = min(start - last, 0.25)
            last = start
            try:
                with self.lock:
                    size = self.size
                    state = dict(self.state)
                if size != self._frame_size:
                    self._allocate(*size)
                frame = self._render(state, start, dt)
                with self.lock:
                    self._frame = frame
                self._ready.set()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.failed = f"{type(e).__name__}: {e}"
                print(f"stage3d: render failed ({self.failed})", flush=True)
                with self.lock:
                    self._frame = None
                return
            took = time.time() - start
            self.render_ms = self.render_ms * 0.9 + took * 100.0
            time.sleep(max(0.0, period - took))

    def _build(self):
        import moderngl
        ctx = self.ctx
        self.mgl = moderngl
        self.sky_prog = ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        self.cloud_prog = ctx.program(vertex_shader=CLOUD_VS, fragment_shader=CLOUD_FS)
        self.mesh_prog = ctx.program(vertex_shader=MESH_VS, fragment_shader=MESH_FS)
        self.ring_prog = ctx.program(vertex_shader=RING_VS, fragment_shader=RING_FS)
        self.present_prog = ctx.program(vertex_shader=SKY_VS, fragment_shader=PRESENT_FS)
        self.sprite_prog = ctx.program(vertex_shader=SPRITE_VS, fragment_shader=SPRITE_FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.quad_vbo = ctx.buffer(quad.tobytes())

        def quad_vao(prog):
            return ctx.vertex_array(prog, [(self.quad_vbo, "2f", "in_pos")])
        self.sky_vao = quad_vao(self.sky_prog)
        self.cloud_vao = quad_vao(self.cloud_prog)
        self.ring_vao = quad_vao(self.ring_prog)
        self.present_vao = quad_vao(self.present_prog)
        self.sprite_vao = quad_vao(self.sprite_prog)

        self.exposure = 1.0
        self.blank = ctx.texture((1, 1), 4, bytes([0, 0, 0, 0]))
        vol = noise_volume()
        self.noise = ctx.texture3d(vol.shape[:3], 4, vol.tobytes())
        self.noise.repeat_x = self.noise.repeat_y = self.noise.repeat_z = True
        self.noise.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.meshes = {}
        self.textures = {}
        self.sprites = {}
        self._dial_key = None
        self._caption_key = None
        self.layout_ = None
        self.clouds = []
        self.rings = []
        self._last_ring = 0.0
        self._view = {"x": 0.0, "y": 0.0}
        self._knob = 0.0
        self._glow = 0.0
        self._speech = 0.0

    def _upload_mesh(self, name, mesh):
        data = mesh.interleaved()
        vbo = self.ctx.buffer(data.tobytes())
        ibo = self.ctx.buffer(mesh.idx.astype(np.uint32).tobytes())
        vao = self.ctx.vertex_array(
            self.mesh_prog, [(vbo, "3f 3f 2f 1f", "in_pos", "in_normal", "in_uv", "in_ao")],
            index_buffer=ibo, index_element_size=4)
        old = self.meshes.get(name)
        if old is not None:
            for obj in old:
                obj.release()
        self.meshes[name] = (vao, vbo, ibo)

    def _texture(self, name, img, mipmaps=True):
        img = img.convert("RGBA")
        old = self.textures.get(name)
        if old is not None and old.size == img.size:
            old.write(img.tobytes())
            if mipmaps:
                old.build_mipmaps()
            return old
        if old is not None:
            old.release()
        tex = self.ctx.texture(img.size, 4, img.tobytes())
        if mipmaps:
            tex.build_mipmaps()
            tex.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
            tex.anisotropy = 8.0
        else:
            tex.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        self.textures[name] = tex
        return tex

    def _allocate(self, w, h):
        ctx = self.ctx
        mgl = self.mgl
        for name in ("scene_fbo", "resolve_fbo", "out_fbo", "cloud_fbo"):
            fbo = getattr(self, name, None)
            if fbo is not None:
                for a in fbo.color_attachments:
                    a.release()
                if fbo.depth_attachment is not None:
                    fbo.depth_attachment.release()
                fbo.release()
        samples = min(MSAA, ctx.max_samples)
        self.scene_fbo = ctx.framebuffer(
            color_attachments=[ctx.renderbuffer((w, h), 4, samples=samples)],
            depth_attachment=ctx.depth_renderbuffer((w, h), samples=samples))
        resolve_tex = ctx.texture((w, h), 4)
        resolve_tex.filter = (mgl.NEAREST, mgl.NEAREST)
        self.resolve_fbo = ctx.framebuffer(color_attachments=[resolve_tex])
        self.out_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 3)])
        cw, ch = max(16, int(w * CLOUD_SCALE)), max(16, int(h * CLOUD_SCALE))
        cloud_tex = ctx.texture((cw, ch), 4, dtype="f2")
        cloud_tex.filter = (mgl.LINEAR, mgl.LINEAR)
        cloud_tex.repeat_x = cloud_tex.repeat_y = False
        self.cloud_fbo = ctx.framebuffer(color_attachments=[cloud_tex])
        self._frame_size = (w, h)
        self.layout_ = Layout(w, h)
        self._build_scene(self.layout_)
        self._texture("stars", constellation_texture(w, h), mipmaps=False)
        self._dial_key = self._caption_key = None
        self.sprites = {}
        lay = self.layout_
        # The small print, at the foot of the right bay.
        footer = Image.fromarray(dw.footer_image(h))
        # A soft dark halo so it reads over a white cloud.
        halo = footer.split()[3].filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3))
        shade = Image.new("RGBA", footer.size, (24, 10, 34, 0))
        shade.putalpha(halo.point(lambda v: min(255, int(v * 1.6))))
        footer = Image.alpha_composite(shade, footer)
        bay_px = lay.bay_width * h
        if lay.props and footer.width > bay_px * 0.96:
            k = bay_px * 0.96 / footer.width
            footer = footer.resize((max(1, int(footer.width * k)), max(1, int(footer.height * k))),
                                   Image.LANCZOS)
        if lay.props:
            fx = int(lay.to_px(lay.bay_centre, 0)[0] - footer.width / 2)
        else:
            fx = w - footer.width - 10
        self._sprite("footer", footer, fx, h - footer.height - 8)

    # -- the set ---------------------------------------------------------------
    def _build_scene(self, lay):
        rng = np.random.default_rng(11)
        r = lay.pillar_r
        bottom = -0.50
        top = 0.497             # the abacus meets the top of the screen

        # Pillars: a lathe profile, fluted along the shaft.
        prof = []
        base_h = 0.075
        cap_h = 0.07

        def add(rad, y):
            prof.append((rad * r, y))
        # torus mouldings at the foot
        for k in range(10):
            t = k / 9
            add(1.32 + 0.12 * math.sin(t * math.pi), bottom + 0.02 + base_h * 0.45 * t)
        for k in range(8):
            t = k / 7
            add(1.12 + 0.1 * math.sin(t * math.pi), bottom + 0.02 + base_h * (0.45 + 0.4 * t))
        shaft0 = bottom + 0.02 + base_h
        shaft1 = top - cap_h
        shaft_rows = 60
        for k in range(shaft_rows):
            t = k / (shaft_rows - 1)
            add(1.0 - 0.08 * t * t, shaft0 + (shaft1 - shaft0) * t)
        # neck ring, echinus
        for k in range(6):
            t = k / 5
            add(0.95 + 0.08 * math.sin(t * math.pi), shaft1 + 0.012 * t)
        for k in range(10):
            t = k / 9
            add(0.95 + 0.42 * (t ** 0.7), shaft1 + 0.012 + (cap_h * 0.55) * t)
        prof.append((0.0001, shaft1 + 0.012 + cap_h * 0.55))
        shaft_start, shaft_end = 18, 18 + shaft_rows
        flutes = 20

        def flute_radius(i, y, th, rr):
            if shaft_start + 1 <= i < shaft_end - 1:
                groove = np.clip(np.cos(th * flutes), 0, 1) ** 0.6
                return rr * (1.0 - 0.06 * groove)
            return rr

        def flute_ao(i, y, th):
            if shaft_start + 1 <= i < shaft_end - 1:
                groove = np.clip(np.cos(th * flutes), 0, 1) ** 0.6
                return 1.0 - 0.45 * groove
            return np.ones_like(th)
        column = lathe(prof, segments=flutes * 10, radius_fn=flute_radius, ao_fn=flute_ao)
        plinth = box(-1.5 * r, 1.5 * r, bottom - 0.2, bottom + 0.022, -1.5 * r, 1.5 * r)
        abacus_y = shaft1 + 0.012 + cap_h * 0.55
        abacus = box(-1.48 * r, 1.48 * r, abacus_y, abacus_y + cap_h * 0.32, -1.48 * r, 1.48 * r)
        pillar = Mesh.merge([column, plinth, abacus])
        self._upload_mesh("pillar", pillar)
        self.pillar_top = abacus_y + cap_h * 0.32
        self.pillar_z = -r * 0.25
        self._upload_mesh("orb", sphere(1.0, 24, 32))

        # The radio, on the left.
        avail = lay.aspect / 2 - lay.pillar_x     # screen edge to the pillar's axis
        self.avail = avail
        rz = 0.10
        k = EYE_Z / (EYE_Z - rz)
        rw = min(0.27, avail * 0.84) / k
        rh, rd = rw * 1.18, rw * 0.46
        # Centred so its projected width (turned a little) clears the edge.
        proj = (rw * math.cos(math.radians(21)) + rd * math.sin(math.radians(21))) * k
        self.radio_pos = ((-lay.aspect / 2 + 0.015 + proj / 2) / k, -0.46 / k, rz)
        self.radio_dims = (rw, rh, rd)
        outline = cathedral_outline(rw, rh, rh * 0.48)
        body = extrude_outline(outline, rd, bevel=rw * 0.06, rings=5)
        self._upload_mesh("radio_body", body)
        colour, props = radio_front_textures(outline=outline, body_w=rw, body_h=rh)
        self._texture("radio_colour", colour)
        self._texture("radio_props", props)
        # Front face as its own mesh, so it can carry the decal.
        face_outline = np.asarray(outline) * np.array([0.93, 1.0])
        face_outline[:, 1] = face_outline[:, 1] * 0.95 + rh * 0.025
        face = extrude_outline(face_outline, 0.004, bevel=0.0019, rings=2)
        # Its UVs relative to the body outline, so the decal lines up.
        lo = np.asarray(outline).min(0)
        hi = np.asarray(outline).max(0)
        face.uv = ((face.pos[:, :2] - lo) / (hi - lo)).astype(np.float32)
        face.pos[:, 2] += rd / 2 - 0.0005
        self._upload_mesh("radio_face", face)
        # Dial: a half disc set into the face.
        dial_w = rw * 0.6
        dial_h = dial_w * 0.5
        dz = rd / 2 + 0.0025
        dy0 = rh * 0.215
        self._upload_mesh("dial", Mesh(
            [(-dial_w / 2, dy0, dz), (dial_w / 2, dy0, dz), (dial_w / 2, dy0 + dial_h, dz),
             (-dial_w / 2, dy0 + dial_h, dz)],
            [0, 1, 2, 0, 2, 3], [(0, 0), (1, 0), (1, 1), (0, 1)],
            normals=[(0, 0, 1)] * 4))
        # Knobs: knurled bakelite.
        kprof = []
        for k in range(14):
            t = k / 13
            kprof.append((1.0 - 0.15 * t * t, t * 0.9))
        kprof.append((0.75, 0.95))
        kprof.append((0.0001, 1.0))

        def knurl(i, y, th, rr):
            if 1 <= i <= 11:
                return rr * (1.0 + 0.045 * np.cos(th * 24))
            return rr
        knob = lathe(kprof, segments=96, radius_fn=knurl)
        knob.pos = knob.pos[:, [0, 2, 1]] * np.array([1, 1, 1], np.float32)   # axis to +z
        knob.normals = knob.normals[:, [0, 2, 1]]
        self._upload_mesh("knob", knob)
        self.knobs = [(-0.30 * rw, rh * 0.13, 0.045 * rw), (0.0, rh * 0.13, 0.058 * rw),
                      (0.30 * rw, rh * 0.13, 0.045 * rw)]

        # The radio mast, on the right, with its eye.
        tower = []
        th_ = 0.58 * min(1.0, lay.bay_width / 0.26)
        w0, w1 = 0.062 * th_ / 0.56, 0.012
        levels = 6
        legs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        rad = 0.0042
        for k in range(levels):
            t0, t1 = k / levels, (k + 1) / levels
            y0, y1 = th_ * t0, th_ * t1
            h0 = w0 + (w1 - w0) * t0
            h1 = w0 + (w1 - w0) * t1
            for (sx_, sz_) in legs:
                tower.append(cylinder_between((sx_ * h0, y0, sz_ * h0), (sx_ * h1, y1, sz_ * h1), rad * 1.4))
            for i in range(4):
                a_, b_ = legs[i], legs[(i + 1) % 4]
                tower.append(cylinder_between((a_[0] * h1, y1, a_[1] * h1), (b_[0] * h1, y1, b_[1] * h1), rad))
                tower.append(cylinder_between((a_[0] * h0, y0, a_[1] * h0), (b_[0] * h1, y1, b_[1] * h1), rad * 0.8))
                tower.append(cylinder_between((b_[0] * h0, y0, b_[1] * h0), (a_[0] * h1, y1, a_[1] * h1), rad * 0.8))
        # A platform for the eye.
        tower.append(box(-w1 * 2.2, w1 * 2.2, th_, th_ + 0.008, -w1 * 2.2, w1 * 2.2))
        self._upload_mesh("tower", Mesh.merge(tower))
        self.tower_h = th_
        self.eye_r = 0.044 * min(1.0, lay.bay_width / 0.26)

        # Clouds: far ones across the sky, and two near ones for the props to sit on.
        self.clouds = [Cloud(rng, lay, EYE_Z) for _ in range(11)]
        self.clouds.sort(key=lambda c: c.z)
        self.prop_clouds = []
        if lay.props:
            for side in (-1, 1):
                x = self.radio_pos[0] if side < 0 else lay.bay_centre
                c = Cloud(rng, lay, EYE_Z, near=True, x=x)
                c.z = -0.12
                c.radius = lay.bay_width * 0.66
                c.y = -0.50 if side < 0 else -0.46
                c.density = 34
                self.prop_clouds.append(c)

    # -- per frame -----------------------------------------------------------
    def _sprite(self, name, img, x, y, opacity=1.0):
        tex = self._texture("sprite_" + name, img, mipmaps=False)
        self.sprites[name] = (tex, (x, y, img.width, img.height), opacity)

    def _draw_mesh(self, name, model, material, tint=(1, 1, 1), glow=0.0, tex_a=None,
                   tex_b=None, gaze=(0, 0, 1)):
        prog = self.mesh_prog
        prog["model"].write(mat_bytes(model))
        prog["material"].value = material
        prog["tint"].value = tuple(float(c) for c in tint)
        prog["glow"].value = float(glow)
        if "gaze" in prog:
            prog["gaze"].value = tuple(float(c) for c in gaze)
        if tex_a is not None:
            tex_a.use(1)
        if tex_b is not None:
            tex_b.use(2)
        self.meshes[name][0].render()

    def _render(self, st, now, dt):
        ctx = self.ctx
        mgl = self.mgl
        w, h = self._frame_size
        lay = self.layout_
        a = lay.aspect

        # Head-coupled eye, eased.
        k = 1.0 - math.exp(-dt * 2.0)
        vx = st["viewer"] if st["present"] else 0.5 * math.sin(now * 0.11)
        self._view["x"] += (vx - self._view["x"]) * k
        eye = (self._view["x"] * 0.22, 0.03 + 0.02 * math.sin(now * 0.07), EYE_Z)
        vp = off_axis(a, eye)
        vp_bytes = mat_bytes(vp)

        speaking = 1.0 if st["speaking"] else 0.0
        self._speech += (speaking - self._speech) * (1.0 - math.exp(-dt * 6.0))
        target_glow = max(0.6 if st["on_air"] else 0.0, self._speech, st["static"] * 0.8)
        self._glow += (target_glow - self._glow) * (1.0 - math.exp(-dt * 4.0))

        # 1. Clouds, at half resolution.
        for c in self.clouds:
            c.update(dt)
        self.cloud_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        ctx.disable(mgl.DEPTH_TEST | mgl.CULL_FACE)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = (mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA)
        prog = self.cloud_prog
        prog["viewproj"].write(vp_bytes)
        prog["eye"].value = eye
        prog["time"].value = now % 1000.0
        prog["noise"].value = 0
        self.noise.use(0)
        for c in sorted(self.clouds + self.prop_clouds, key=lambda c: c.z):
            prog["cloud"].value = (float(c.x), float(c.y), float(c.z), float(c.radius))
            puffs = np.zeros((16, 4), np.float32)
            puffs[:len(c.puffs)] = c.puffs
            prog["puffs"].write(puffs.tobytes())
            prog["puff_count"].value = len(c.puffs)
            prog["base_y"].value = float(c.base)
            prog["seed"].value = float(c.seed)
            prog["density"].value = float(c.density)
            self.cloud_vao.render(mgl.TRIANGLE_STRIP)

        # 2. Sky (with the clouds over it) and the set, multisampled.
        self.scene_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.disable(mgl.BLEND)
        prog = self.sky_prog
        prog["eye"].value = eye
        prog["aspect"].value = a
        prog["time"].value = now % 1000.0
        prog["screen"].value = (float(w), float(h))
        _set(prog, "exposure", self.exposure)
        sx, sy, s_px, _ = lay.pic_px
        prog["sun_px"].value = (sx + s_px / 2.0, h - (sy + s_px * 0.42))
        self.cloud_fbo.color_attachments[0].use(0)
        prog["clouds"].value = 0
        self.textures["stars"].use(1)
        prog["stars_tex"].value = 1
        self.sky_vao.render(mgl.TRIANGLE_STRIP)

        ctx.enable(mgl.DEPTH_TEST)
        prog = self.mesh_prog
        prog["viewproj"].write(vp_bytes)
        prog["eye"].value = eye
        prog["time"].value = now % 1000.0
        _set(prog, "exposure", self.exposure)
        self.noise.use(0)
        self.blank.use(1)
        self.blank.use(2)
        prog["noise"].value = 0
        prog["tex_a"].value = 1
        prog["tex_b"].value = 2

        marble = lin((0.90, 0.58, 0.54))
        if lay.pillars:
            for side in (-1, 1):
                m = translate(side * lay.pillar_x, 0.0, self.pillar_z)
                self._draw_mesh("pillar", m, 0, marble)

        if lay.props:
            self._draw_radio(st, now, dt, lay)
            self._draw_tower(st, now, dt, lay, eye)

        # Broadcast rings from the eye: additive, depth-tested, no depth write.
        if lay.props and self.rings:
            ctx.enable(mgl.BLEND)
            ctx.blend_func = (mgl.ONE, mgl.ONE)
            ctx.depth_mask = False
            prog = self.ring_prog
            prog["viewproj"].write(vp_bytes)
            for ring in self.rings:
                age = now - ring[0]
                life = ring[1]
                t = age / life
                rr = self.eye_r * (1.4 + 5.5 * t)
                prog["ring"].value = (ring[2][0], ring[2][1], ring[2][2] - 0.01, rr)
                prog["alpha"].value = float((1.0 - t) ** 1.5 * ring[3])
                prog["colour"].value = (1.0, 0.78, 0.45)
                prog["soft"].value = 0.05
                self.ring_vao.render(mgl.TRIANGLE_STRIP)
            ctx.depth_mask = True
            ctx.disable(mgl.BLEND)

        # 3. Resolve, flip and swizzle into the output, then the 2D sprites.
        ctx.copy_framebuffer(self.resolve_fbo, self.scene_fbo)
        self.out_fbo.use()
        ctx.disable(mgl.DEPTH_TEST)
        self.resolve_fbo.color_attachments[0].use(0)
        self.present_prog["scene"].value = 0
        self.present_vao.render(mgl.TRIANGLE_STRIP)
        self._update_caption(st, lay)
        if self.sprites:
            ctx.enable(mgl.BLEND)
            ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
            prog = self.sprite_prog
            prog["screen"].value = (float(w), float(h))
            prog["image"].value = 0
            for tex, (x, y, sw, sh), opacity in self.sprites.values():
                tex.use(0)
                prog["rect"].value = (float(x), float(y), float(sw), float(sh))
                prog["opacity"].value = float(opacity)
                self.sprite_vao.render(mgl.TRIANGLE_STRIP)
            ctx.disable(mgl.BLEND)
        data = self.out_fbo.read(components=3, alignment=1)
        return np.frombuffer(data, np.uint8).reshape(h, w, 3)

    def _radio_anchor(self, lay):
        return self.radio_pos

    def _draw_radio(self, st, now, dt, lay):
        rw, rh, rd = self.radio_dims
        x, y, z = self._radio_anchor(lay)
        bob = 0.006 * math.sin(now * 0.9)
        yaw = math.radians(14) + math.radians(7) * math.sin(now * 0.23)
        # A shiver when tuning through static.
        shake = st["static"] * 0.004 * math.sin(now * 61.0)
        model = translate(x + shake, y + bob, z) @ rot_y(yaw) @ rot_x(math.radians(-4))
        self._draw_mesh("radio_body", model, 2)
        self._draw_mesh("radio_face", model, 3, glow=self._glow,
                        tex_a=self.textures["radio_colour"], tex_b=self.textures["radio_props"])
        key = (round(st["freq"], 1), bool(st["on_air"]))
        if key != self._dial_key:
            self._dial_key = key
            self._texture("dial", dial_texture(st["freq"], st["on_air"]))
        self._draw_mesh("dial", model, 4, glow=self._glow, tex_a=self.textures["dial"])
        target = (st["freq"] - dw.DIAL_MIN) * 0.9
        self._knob += (target - self._knob) * (1.0 - math.exp(-dt * 8.0))
        for i, (kx, ky, kr) in enumerate(self.knobs):
            spin = self._knob if i == 1 else (0.6 if i == 0 else -0.4)
            m = model @ translate(kx, ky, rd / 2) @ rot_z(spin) @ scale(kr, kr, kr * 0.55)
            self._draw_mesh("knob", m, 6, lin((0.22, 0.12, 0.07)))

    def _draw_tower(self, st, now, dt, lay, eye):
        base_y = -0.40
        x, z = lay.bay_centre, -0.05
        sway = math.radians(1.5) * math.sin(now * 0.5)
        model = translate(x, base_y, z) @ rot_y(math.radians(30) + 0.1 * math.sin(now * 0.13)) \
            @ rot_z(sway)
        self._draw_mesh("tower", model, 1, lin((1.0, 0.84, 0.55)))
        er = self.eye_r
        ey_ = base_y + self.tower_h + er * 1.05
        centre = np.array([x + math.sin(-sway) * self.tower_h * 0.0, ey_, z])
        # The eye looks at the viewer (the eye point), with the odd glance away.
        look = np.array(eye) - centre
        look[0] += 0.4 * math.sin(now * 0.31) * (0 if st["present"] else 1)
        look /= np.linalg.norm(look)
        m = translate(*centre) @ scale(er)
        self._draw_mesh("orb", m, 5, glow=self._glow, gaze=tuple(look))
        # Rings: slow and few normally, fast while on air or speaking.
        interval = 0.7 if (st["on_air"] or self._speech > 0.5) else 2.6
        if now - self._last_ring > interval:
            self._last_ring = now
            strength = 1.0 if interval < 1.0 else 0.45
            self.rings.append((now, 2.4, (float(centre[0]), float(centre[1]), float(centre[2])),
                               strength))
        self.rings = [r for r in self.rings if now - r[0] < r[1]]

    def _update_caption(self, st, lay):
        if not lay.props:
            return
        title = f"“{st['heard']}”" if st["on_air"] and st["heard"] else st["name"]
        sub = f"{st['freq']:.1f} MHz"
        key = (title, sub, bool(st["on_air"]))
        if key == self._caption_key:
            return
        self._caption_key = key
        w, h = self._frame_size
        width = (lay.aspect / 2 - lay.pillar_x - lay.pillar_r * 1.6) * h
        img = caption_image(width, title, sub, st["on_air"])
        rx, ry, rz = self._radio_anchor(lay)
        top = ry + self.radio_dims[1] * 1.04
        k = EYE_Z / (EYE_Z - rz)
        cx, cy = lay.to_px(rx * k, top * k)
        x = int(max(6, (lay.to_px(-lay.pillar_x - lay.pillar_r * 1.6, 0)[0] - img.width) / 2))
        y = int(max(4, cy - img.height - lay.h * 0.015))
        self._sprite("caption", img, x, y)


def try_start(w, h):
    stage = Stage3D()
    stage.start(w, h)
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if stage.failed:
            return None
        if stage._ready.is_set():
            return stage
        time.sleep(0.02)
    print("stage3d: took too long to start; using the flat stage", flush=True)
    stage.close()
    return None
