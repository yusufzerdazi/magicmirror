"""On-screen settings panel, driven from the keyboard.

The `mischief 3.0` frontend toggles a settings panel with `c` and exposes the
inference parameters and a model switch; this is the same thing for a build with
no mouse and no DOM. It is drawn with PIL rather than OpenCV's own text so it
can have real type, rounded corners and translucency — closer to the panel it
stands in for than Hershey strokes on a grey box.

Rows are selected with the arrow keys (or j/k), values changed with left/right
(or h/l), and the model is applied with return.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import models
import upscale


def app_edge_styles():
    import app
    return app.EDGE_STYLES

# waitKeyEx codes. macOS HighGUI reports arrows in the NSEvent function-key
# range; the Linux/Qt values are accepted too so the same build works anywhere.
KEY_UP = (63232, 82, 2490368, 65362, 0, 0x100000 | 82)
KEY_DOWN = (63233, 84, 2621440, 65364, 1, 0x100000 | 84)
KEY_LEFT = (63234, 81, 2424832, 65361, 2, 0x100000 | 81)
KEY_RIGHT = (63235, 83, 2555904, 65363, 3, 0x100000 | 83)
KEY_RETURN = (13, 10)
KEY_ESCAPE = (27,)

# Letter aliases, for builds where HighGUI never delivers the arrow keys.
ALIAS_UP = tuple(ord(c) for c in "kw")
ALIAS_DOWN = tuple(ord(c) for c in "js")
ALIAS_LEFT = tuple(ord(c) for c in "ha,")
ALIAS_RIGHT = tuple(ord(c) for c in "ld.")

FONT_CANDIDATES = ("/System/Library/Fonts/SFNS.ttf",
                   "/System/Library/Fonts/Helvetica.ttc")

# Palette, in the piece's own colours.
ACCENT = (34, 161, 236)          # #22A1EC, the coin's blue
ACCENT_WARM = (255, 186, 241)    # #FFBAF1, the coin's pink
PANEL_BG = (12, 13, 18, 226)
PANEL_EDGE = (255, 255, 255, 38)
# Pre-blended against PANEL_BG: a fill replaces pixels, it does not composite.
ROW_SELECTED = (38, 40, 46, 240)
TEXT = (255, 255, 255, 255)
TEXT_DIM = (255, 255, 255, 150)
TEXT_FAINT = (255, 255, 255, 95)
TRACK = (58, 60, 68, 240)


class Field:
    """One row: a label, a way to read the value, and a way to nudge it."""

    def __init__(self, label, get, adjust, fmt=str, hint="", apply=None,
                 span=None, options=None):
        self.label = label
        self.get = get
        self.adjust = adjust
        self.fmt = fmt
        self.hint = hint
        self.apply = apply
        self.span = span          # (low, high) draws a slider track
        self.options = options    # a list draws a position dot row

    def fraction(self):
        """Where this value sits in its range, for the slider, or None."""
        if self.span:
            low, high = self.span
            value = float(self.get())
            return min(max((value - low) / (high - low), 0.0), 1.0)
        return None


def _clamp(value, low, high):
    return min(max(value, low), high)


class ConfigPanel:
    """Keyboard-driven settings overlay."""

    def __init__(self, state):
        self.state = state
        self.visible = False          # opened with `c`
        self.selected = 0
        self.fields = self._build()
        self.message = ""
        self.unknown_key = None
        self._fonts = {}
        self._cache = None
        self._cache_key = None

    # -- fields --------------------------------------------------------------
    def _build(self):
        st = self.state
        names = list(models.AVAILABLE)

        def cycle_model(delta):
            current = st.pending_model or st.manager.name
            index = (names.index(current) + delta) % len(names)
            st.pending_model = names[index]

        def apply_model():
            target = st.pending_model
            if not target or target == st.manager.name:
                st.pending_model = None
                return "already loaded"
            if st.switch_model(target):
                return f"loading {target}"
            return "busy"

        fade_steps = [None, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]
        return [
            Field("model", lambda: st.pending_model or st.manager.name, cycle_model,
                  hint="return to load", apply=apply_model, options=names),
            Field("steps", lambda: st.steps,
                  lambda d: setattr(st, "steps", int(_clamp(st.steps + d, 1, 16))),
                  span=(1, 16)),
            Field("strength", lambda: st.strength,
                  lambda d: setattr(st, "strength",
                                    round(_clamp(st.strength + 0.025 * d, 0.05, 1.0), 3)),
                  fmt=lambda v: f"{v:.3f}", span=(0.05, 1.0)),
            Field("seed", lambda: st.seed, lambda d: st.set_seed(st.seed + d)),
            Field("blend", lambda: st.prompt_state.describe()[2],
                  lambda d: st.prompt_state.set_blend(
                      st.prompt_state.describe()[2] + 0.05 * d),
                  fmt=lambda v: f"{v:.2f}", span=(0.0, 1.0)),
            Field("cross-fade", lambda: "auto" if st.fade_step is None else st.fade_step,
                  st.adjust_fade,
                  fmt=lambda v: v if isinstance(v, str) else f"{v:.2f}",
                  options=fade_steps),
            Field("auto-switch", lambda: st.rotate_minutes,
                  st.adjust_rotation,
                  fmt=lambda v: "off" if not v else f"{v} min",
                  hint="cycle through the models on this interval",
                  options=[0, 5, 10, 15, 30, 45, 60]),
            Field("edges", lambda: st.edge_style, st.cycle_edges,
                  hint="control image for the ControlNet models",
                  options=list(st.edge_styles())),
            Field("framing", lambda: st.capture_fit, st.cycle_framing,
                  options=["wide", "stretch", "crop"]),
            Field("frame", lambda: st.frame, st.cycle_frame,
                  hint="16:10 crops a full-width band from the square",
                  options=["square", "16:10"]),
            Field("crop position", lambda: st.crop_bias, st.adjust_crop_bias,
                  fmt=lambda v: f"{v:.2f}", span=(0.0, 1.0),
                  hint="where the 16:10 band sits: 0 top, 1 bottom"),
            Field("upscaler", lambda: st.upscaler, st.cycle_upscaler,
                  hint="upscale to 1080p before display",
                  options=list(upscale.KINDS)),
            Field("rotation", lambda: st.rotation,
                  lambda d: st.set_rotation((st.rotation + 90 * d) % 360),
                  fmt=lambda v: f"{v}°", options=[0, 90, 180, 270]),
            Field("dream world", lambda: st.world_mode, st.cycle_world,
                  hint="auto alternates mirror and world each station",
                  options=["auto", "on", "off"]),
            Field("kinect", lambda: ("on" if st.kinect_on else "off")
                  if st.has_kinect else "none", st.toggle_kinect,
                  hint="k toggles; off = camera only, for depth and cut-outs",
                  options=["on", "off"]),
            Field("kinect near", lambda: st.depth_near, lambda d: st.adjust_depth(d, 0),
                  fmt=lambda v: f"{v} mm", span=(300, 2000),
                  hint="nearest edge of the working volume"),
            Field("kinect far", lambda: st.depth_far, lambda d: st.adjust_depth(0, d),
                  fmt=lambda v: f"{v} mm", span=(1000, 6000),
                  hint="bring in until the back wall disappears"),
            Field("tint", lambda: "on" if st.tint_on else "off",
                  lambda d: st.toggle_tint(), options=["off", "on"]),
            Field("background", lambda: "circles" if st.background_on else "black",
                  lambda d: st.toggle_background(), options=["black", "circles"]),
        ]

    # -- input ---------------------------------------------------------------
    def handle(self, key):
        """Return True if the key was consumed by the panel."""
        if not self.visible:
            return False
        if key in KEY_ESCAPE or key == ord("c"):
            self.visible = False
            self.message = ""
            return True
        if key in KEY_DOWN or key in ALIAS_DOWN:
            self.selected = (self.selected + 1) % len(self.fields)
            self.unknown_key = None
        elif key in KEY_UP or key in ALIAS_UP:
            self.selected = (self.selected - 1) % len(self.fields)
            self.unknown_key = None
        elif key in KEY_LEFT or key in ALIAS_LEFT:
            self.fields[self.selected].adjust(-1)
            self.unknown_key = None
        elif key in KEY_RIGHT or key in ALIAS_RIGHT:
            self.fields[self.selected].adjust(+1)
            self.unknown_key = None
        elif key in KEY_RETURN:
            field = self.fields[self.selected]
            if field.apply is not None:
                self.message = field.apply() or ""
            self.unknown_key = None
        else:
            # Surfaced in the footer: if the arrows are not arriving as any code
            # we know, this is how you find out what they do arrive as.
            self.unknown_key = key
        return True     # swallow everything else while open

    # -- drawing -------------------------------------------------------------
    def _font(self, size, weight="Regular"):
        key = (size, weight)
        cached = self._fonts.get(key)
        if cached is not None:
            return cached
        font = None
        for path in FONT_CANDIDATES:
            if not Path(path).exists():
                continue
            try:
                font = ImageFont.truetype(path, size)
                try:
                    font.set_variation_by_name(weight)
                except Exception:
                    pass        # not a variable font; the regular cut will do
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        self._fonts[key] = font
        return font

    def _state_key(self):
        """Everything the drawing depends on, so it is only redrawn on change."""
        manager = self.state.manager
        return (self.selected, self.message, manager.name, manager.loading,
                manager.error, self.unknown_key,
                tuple(str(f.get()) for f in self.fields))

    def _render(self, width):
        """Draw the panel to an RGBA image."""
        s = max(0.62, width / 1900.0)          # scale against display width
        pad = int(22 * s)
        row_h = int(40 * s)
        head_h = int(52 * s)
        foot_h = int(40 * s)
        panel_w = int(400 * s)
        panel_h = head_h + row_h * len(self.fields) + foot_h + pad

        img = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        radius = int(18 * s)
        d.rounded_rectangle((0, 0, panel_w - 1, panel_h - 1), radius=radius,
                            fill=PANEL_BG, outline=PANEL_EDGE, width=max(1, int(s)))

        f_title = self._font(int(15 * s), "Semibold")
        f_label = self._font(int(14 * s), "Regular")
        f_value = self._font(int(14 * s), "Medium")
        f_small = self._font(int(11 * s), "Regular")

        d.text((pad, int(19 * s)), "SETTINGS", font=f_title, fill=TEXT_DIM)
        d.text((panel_w - pad, int(20 * s)), "c", font=f_small, fill=TEXT_FAINT,
               anchor="ra")
        line_y = head_h - int(10 * s)
        d.line((pad, line_y, panel_w - pad, line_y), fill=(255, 255, 255, 22),
               width=max(1, int(s)))

        value_x = panel_w - pad
        y = head_h
        for index, field in enumerate(self.fields):
            selected = index == self.selected
            if selected:
                d.rounded_rectangle((int(8 * s), y + int(2 * s),
                                     panel_w - int(8 * s), y + row_h - int(4 * s)),
                                    radius=int(9 * s), fill=ROW_SELECTED)
                d.rounded_rectangle((int(8 * s), y + int(7 * s),
                                     int(11 * s), y + row_h - int(9 * s)),
                                    radius=int(2 * s), fill=ACCENT + (255,))

            value = field.get()
            text_value = value if isinstance(value, str) else field.fmt(value)
            label_colour = TEXT if selected else TEXT_DIM
            d.text((pad, y + int(11 * s)), field.label, font=f_label,
                   fill=label_colour)
            d.text((value_x, y + int(11 * s)), str(text_value), font=f_value,
                   fill=TEXT if selected else (255, 255, 255, 190), anchor="ra")

            fraction = field.fraction()
            if fraction is not None:
                track_y = y + row_h - int(9 * s)
                track_x1 = panel_w - pad
                track_x0 = track_x1 - int(120 * s)
                d.rounded_rectangle((track_x0, track_y, track_x1, track_y + int(3 * s)),
                                    radius=int(2 * s), fill=TRACK)
                filled = track_x0 + (track_x1 - track_x0) * fraction
                if filled > track_x0:
                    d.rounded_rectangle((track_x0, track_y, filled,
                                         track_y + int(3 * s)),
                                        radius=int(2 * s),
                                        fill=ACCENT + (235 if selected else 150,))
            elif field.options and len(field.options) <= 8:
                # A row of dots showing where this option sits in its list.
                try:
                    position = field.options.index(
                        value if not isinstance(value, str) else value)
                except ValueError:
                    position = -1
                dot_r = max(1, int(2.5 * s))
                gap = int(9 * s)
                x = panel_w - pad - (len(field.options) - 1) * gap
                for i in range(len(field.options)):
                    cx = x + i * gap
                    cy = y + row_h - int(7 * s)
                    on = i == position
                    d.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r),
                              fill=(ACCENT + (240,)) if on else (255, 255, 255, 55))
            y += row_h

        manager = self.state.manager
        if not manager.loading and self.message.startswith("loading"):
            self.message = f"{manager.name} loaded"   # the swap finished
        if manager.loading:
            status, colour = f"loading {manager.loading}…", ACCENT_WARM + (255,)
        elif manager.error:
            status, colour = manager.error[:46], (255, 140, 140, 255)
        elif self.unknown_key is not None:
            status = f"unrecognised key {self.unknown_key} — use j/k and h/l"
            colour = ACCENT_WARM + (255,)
        else:
            hint = self.fields[self.selected].hint
            status = self.message or hint or "j/k or ↑↓ select   h/l or ←→ change"
            colour = TEXT_FAINT
        d.text((pad, y + int(10 * s)), status, font=f_small, fill=colour)
        return img

    def draw(self, surface_bgr):
        if not self.visible:
            return
        H, W = surface_bgr.shape[:2]
        key = (self._state_key(), W)
        if key != self._cache_key:
            self._cache = self._render(W)
            self._cache_key = key
        panel = np.asarray(self._cache)
        ph, pw = panel.shape[:2]
        margin = int(max(18, W * 0.016))
        x0, y0 = W - pw - margin, margin
        if x0 < 0 or y0 + ph > H:
            x0, y0 = max(0, x0), max(0, min(y0, H - ph))
        x1, y1 = min(W, x0 + pw), min(H, y0 + ph)
        sub = panel[:y1 - y0, :x1 - x0]
        alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
        rgb = sub[:, :, :3][:, :, ::-1].astype(np.float32)     # RGBA -> BGR
        region = surface_bgr[y0:y1, x0:x1].astype(np.float32)
        surface_bgr[y0:y1, x0:x1] = np.clip(
            region * (1.0 - alpha) + rgb * alpha, 0, 255).astype(np.uint8)
