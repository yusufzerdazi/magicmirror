"""Simulated camera and Kinect, for working on the mirror without the hardware.

One scripted scene is seen by both sensors, so the colour frame and the depth
frame agree the way a real camera sitting on a Kinect would: people walk in,
dance, lean towards the mirror, are joined by a friend and wander off again,
with stretches of empty room in between. That exercises everything that
depends on who is in front of the mirror — the depth cut-out, presence and
arrival detection, Dream World compositing — not just the diffusion loop.

    app.py --camera sim --depth sim      # both
    ./start.sh --simulate                # the same, with the event flags

The depth side imitates the Kinect v1's faults that the real pipeline has to
cope with: a band of no-reading "shadow" beside each person (the IR projector
and camera are offset), speckled holes along silhouettes, and a floor that
creeps into the near end of the working volume.
"""

import math
import threading
import time

import cv2
import numpy as np

CAMERA_SIZE = (1280, 720)
KINECT_SIZE = (640, 480)
FPS = 30.0

# One loop of the script, seconds.
LOOP_SECONDS = 72.0
WALL_MM = 4200.0          # behind the default 3000mm far edge, so it drops out


def _ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lerp(a, b, t):
    return a + (b - a) * t


class Person:
    """A figure on a schedule: walk in, stay, walk out."""

    def __init__(self, enter, leave, side, home_x, home_z, shirt, skin, hair,
                 height=1.72, dance=1.0, lean=None):
        self.enter, self.leave = enter, leave
        self.side = side              # -1 enters from the left, +1 from the right
        self.home_x, self.home_z = home_x, home_z
        self.shirt, self.skin, self.hair = shirt, skin, hair
        self.height = height
        self.dance = dance
        self.lean = lean              # (start, end) of a lean towards the mirror

    def state(self, t):
        """(x metres, z millimetres, phase) or None when off stage."""
        walk = 4.0
        if t < self.enter or t > self.leave:
            return None
        off_x = self.side * 2.6
        if t < self.enter + walk:
            k = _ease((t - self.enter) / walk)
            x = _lerp(off_x, self.home_x, k)
        elif t > self.leave - walk:
            k = _ease((t - (self.leave - walk)) / walk)
            x = _lerp(self.home_x, -off_x, k)
        else:
            x = self.home_x + 0.18 * math.sin(t * 0.9 + self.home_x)
        z = self.home_z
        if self.lean is not None and self.lean[0] <= t <= self.lean[1]:
            span = self.lean[1] - self.lean[0]
            k = math.sin(math.pi * (t - self.lean[0]) / span)
            z -= 900.0 * k
        return x, z, t


SCRIPT = (
    Person(4.0, 52.0, -1, -0.15, 1900.0, (60, 70, 200), (150, 180, 220),
           (30, 30, 40), lean=(16.0, 24.0)),
    Person(30.0, 66.0, +1, 0.55, 2300.0, (190, 160, 40), (90, 120, 170),
           (20, 20, 25), height=1.64, dance=1.4),
)


class Scene:
    """The script, rendered for either sensor."""

    def __init__(self, people=SCRIPT, loop=LOOP_SECONDS, clock=time.time):
        self.people = people
        self.loop = loop
        self.clock = clock
        self.t0 = clock()
        self.rng = np.random.default_rng(7)
        self._room = {}

    def t(self):
        return (self.clock() - self.t0) % self.loop

    def occupants(self, t=None):
        t = self.t() if t is None else t
        return sum(1 for p in self.people if p.state(t) is not None)

    # -- geometry ------------------------------------------------------------
    @staticmethod
    def _project(x_m, y_m, z_mm, size, fov_deg):
        w, h = size
        f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        z = z_mm / 1000.0
        return w / 2.0 + f * x_m / z, h * 0.55 - f * y_m / z, f / z

    def _limbs(self, person, t):
        """Joint positions in metres (x right, y up from the camera height)."""
        x, z, phase = person.state(t)
        hgt = person.height
        floor = -1.15                         # camera at ~1.15m
        sway = 0.06 * math.sin(phase * 2.2 * person.dance)
        bounce = 0.03 * abs(math.sin(phase * 2.2 * person.dance))
        hip = (x + sway * 0.5, floor + hgt * 0.53 + bounce)
        neck = (x + sway, floor + hgt * 0.83 + bounce)
        head = (neck[0], neck[1] + 0.13)
        arms = []
        for s in (-1, 1):
            shoulder = (neck[0] + s * 0.19, neck[1] - 0.04)
            a = math.sin(phase * 3.1 * person.dance + (0 if s < 0 else 1.7))
            up = 0.5 + 0.5 * a
            elbow = (shoulder[0] + s * 0.28 * math.cos(math.radians(-70 + 120 * up)),
                     shoulder[1] + 0.28 * math.sin(math.radians(-70 + 120 * up)))
            wave = math.sin(phase * 6.0 + s) * 0.6
            hand = (elbow[0] + s * 0.26 * math.cos(math.radians(-40 + 130 * up) + wave * 0.4),
                    elbow[1] + 0.26 * math.sin(math.radians(-40 + 130 * up) + wave * 0.4))
            arms.append((shoulder, elbow, hand))
        legs = []
        for s in (-1, 1):
            step = 0.08 * math.sin(phase * 2.2 * person.dance + (0 if s < 0 else math.pi))
            legs.append(((hip[0] + s * 0.1, hip[1]),
                         (hip[0] + s * 0.12 + step, floor + 0.02)))
        return dict(x=x, z=z, hip=hip, neck=neck, head=head, arms=arms, legs=legs)

    def _draw_person(self, img, person, t, size, fov, value_fn, detail):
        j = self._limbs(person, t)
        z = j["z"]

        def P(pt, dz=0.0):
            px, py, scale = self._project(pt[0], pt[1], z + dz, size, fov)
            return (int(round(px)), int(round(py))), scale

        def limb(a, b, width_m, colour, dz=0.0):
            (pa, s), (pb, _) = P(a, dz), P(b, dz)
            cv2.line(img, pa, pb, colour, max(2, int(width_m * s)), cv2.LINE_AA)

        shirt = value_fn("shirt", z)
        # Legs, then torso, then arms (in front, slightly nearer), then head.
        for hip, foot in j["legs"]:
            limb(hip, foot, 0.14, value_fn("legs", z + 40))
        (n, s) = P(j["neck"])
        (hp, _) = P(j["hip"])
        torso = np.array([
            (n[0] - 0.21 * s, n[1]), (n[0] + 0.21 * s, n[1]),
            (hp[0] + 0.17 * s, hp[1]), (hp[0] - 0.17 * s, hp[1])], np.int32)
        cv2.fillConvexPoly(img, torso, shirt, cv2.LINE_AA)
        for shoulder, elbow, hand in j["arms"]:
            limb(shoulder, elbow, 0.10, value_fn("shirt", z - 60), dz=-60)
            limb(elbow, hand, 0.08, value_fn("skin", z - 120), dz=-120)
            (ph, sh) = P(hand, -120)
            cv2.circle(img, ph, max(2, int(0.05 * sh)), value_fn("skin", z - 120), -1,
                       cv2.LINE_AA)
        (hc, s) = P(j["head"], -30)
        axes = (max(3, int(0.095 * s)), max(4, int(0.125 * s)))
        cv2.ellipse(img, hc, axes, 0, 0, 360, value_fn("skin", z - 30), -1, cv2.LINE_AA)
        if detail:
            hair = value_fn("hair", z)
            cv2.ellipse(img, (hc[0], hc[1] - axes[1] // 3), (axes[0] + 2, axes[1] // 2 + 4),
                        0, 180, 360, hair, -1, cv2.LINE_AA)
            for sx in (-1, 1):
                cv2.circle(img, (hc[0] + sx * axes[0] // 2 - sx, hc[1]),
                           max(1, axes[0] // 6), (25, 25, 25), -1, cv2.LINE_AA)
            cv2.ellipse(img, (hc[0], hc[1] + axes[1] // 2), (axes[0] // 3, axes[1] // 8),
                        0, 0, 180, (40, 40, 120), max(1, axes[0] // 10), cv2.LINE_AA)
        return j

    # -- colour camera -------------------------------------------------------
    def _room_colour(self, size, t):
        w, h = size
        key = size
        if key not in self._room:
            ys = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
            wall = np.array((38, 34, 40), np.float32) * (1.0 - 0.3 * ys)
            room = np.broadcast_to(wall, (h, w, 3)).copy()
            floor_y = int(h * 0.78)
            room[floor_y:] = (28, 30, 34)
            # A few fixtures on the wall, for the edge detector to find.
            cv2.rectangle(room, (int(w * 0.07), int(h * 0.18)), (int(w * 0.2), int(h * 0.5)),
                          (70, 60, 90), -1)
            cv2.rectangle(room, (int(w * 0.8), int(h * 0.12)), (int(w * 0.93), int(h * 0.44)),
                          (50, 80, 70), -1)
            self._room[key] = room
        room = self._room[key].copy()
        # Party lights sweeping the wall.
        for i, colour in enumerate(((200, 60, 220), (40, 160, 255), (60, 220, 120))):
            cx = int(w * (0.5 + 0.42 * math.sin(t * (0.35 + 0.1 * i) + i * 2)))
            cy = int(h * (0.25 + 0.1 * math.cos(t * 0.5 + i)))
            blob = np.zeros((h, w), np.float32)
            cv2.circle(blob, (cx, cy), int(h * 0.22), 1.0, -1)
            blob = cv2.GaussianBlur(blob, (0, 0), h * 0.08)[:, :, None] * 0.35
            room = room + blob * np.array(colour, np.float32)
        return room

    def camera_frame(self, t=None):
        t = self.t() if t is None else t
        img = self._room_colour(CAMERA_SIZE, t)
        ordered = sorted((p for p in self.people if p.state(t) is not None),
                         key=lambda p: -p.state(t)[1])
        for p in ordered:
            palette = {"shirt": p.shirt, "skin": p.skin, "hair": p.hair,
                       "legs": (40, 40, 50)}
            self._draw_person(img, p, t, CAMERA_SIZE, 70.0,
                              lambda part, z, pal=palette: tuple(int(c) for c in pal[part]),
                              detail=True)
        noise = self.rng.normal(0, 4.0, img.shape).astype(np.float32)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    # -- Kinect --------------------------------------------------------------
    def depth_frame(self, t=None):
        """Millimetres, 0 = no reading, at the Kinect v1's 640x480."""
        t = self.t() if t is None else t
        w, h = KINECT_SIZE
        depth = np.full((h, w), WALL_MM, np.float32)
        floor_y = int(h * 0.62)
        # The floor recedes from ~1.2m at the bottom of the frame to the wall.
        rows = np.linspace(0, 1, h - floor_y, dtype=np.float32)[:, None]
        depth[floor_y:] = np.minimum(WALL_MM, 1200.0 / np.maximum(rows, 0.29))
        ordered = sorted((p for p in self.people if p.state(t) is not None),
                         key=lambda p: -p.state(t)[1])
        for p in ordered:
            layer = np.zeros((h, w), np.float32)
            self._draw_person(layer, p, t, KINECT_SIZE, 58.0,
                              lambda part, z: float(z), detail=False)
            # Depth-tested: a person cannot cover floor that is nearer than
            # they are.
            body = (layer > 0) & ((depth <= 0) | (layer < depth))
            # IR shadow: the projector sits to one side of the camera, so a
            # band beside each person gets no reading.
            shadow = np.zeros_like(body)
            shift = max(2, int(14 * 1900.0 / max(400.0, p.state(t)[1])))
            shadow[:, shift:] = body[:, :-shift]
            depth[shadow & ~body] = 0.0
            depth[body] = layer[body]
        # Speckled holes along edges and a little sensor noise.
        edges = cv2.Canny((np.clip(depth / WALL_MM, 0, 1) * 255).astype(np.uint8), 20, 60) > 0
        holes = edges & (self.rng.random((h, w)) < 0.35)
        depth[holes] = 0.0
        depth[depth > 0] += self.rng.normal(0, 6.0, int((depth > 0).sum()))
        return depth


_SCENE = []
_SCENE_LOCK = threading.Lock()


def scene():
    """The one shared scene, so camera and Kinect see the same people."""
    with _SCENE_LOCK:
        if not _SCENE:
            _SCENE.append(Scene())
        return _SCENE[0]


class SimVideoCapture:
    """Stands in for cv2.VideoCapture: same read()/isOpened()/release()."""

    def __init__(self, scene_=None, fps=FPS):
        self.scene = scene_ or scene()
        self.interval = 1.0 / fps
        self._next = time.time()
        self._open = True

    def isOpened(self):
        return self._open

    def read(self):
        delay = self._next - time.time()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next + self.interval, time.time())
        return True, self.scene.camera_frame()

    def release(self):
        self._open = False


def kinect_reader(scene_=None, fps=FPS):
    """(read, close) pair for depth.DepthSource, paced like the real sensor."""
    sc = scene_ or scene()
    state = {"next": time.time()}

    def read():
        delay = state["next"] - time.time()
        if delay > 0:
            time.sleep(delay)
        state["next"] = max(state["next"] + 1.0 / fps, time.time())
        return sc.depth_frame()

    return read, (lambda: None)


if __name__ == "__main__":
    # Contact sheet of the script, for checking the simulation itself.
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sim-sheet.png"
    sc = Scene()
    tiles = []
    for t in (2.0, 8.0, 20.0, 36.0, 45.0, 58.0):
        cam = cv2.resize(sc.camera_frame(t), (480, 270))
        d = sc.depth_frame(t)
        vis = np.where((d > 600) & (d < 3000), 255 - (d - 600) / 2400 * 230, 0)
        vis = cv2.resize(cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_GRAY2BGR), (360, 270))
        cv2.putText(cam, f"t={t:.0f}s occupants={sc.occupants(t)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(np.hstack([cam, vis]))
    cv2.imwrite(out, np.vstack(tiles))
    print(out)
