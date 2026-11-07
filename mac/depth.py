"""Kinect depth as the ControlNet's line drawing.

Depth is a much better source for this than the colour camera. The sketch
ControlNet wants clean contours, and depth discontinuities give exactly that:
the outline of whoever is standing there, independent of how the room is lit.
The colour camera needs equalising and still produces nearly nothing in a dim
room, which is what made early attempts look untethered.

Both generations are supported, since they need entirely different stacks:

* Kinect v1 (Xbox 360) through `freenect` — `brew install libfreenect`, then
  build its Python bindings (see the README).
* Kinect v2 (Xbox One) through `pylibfreenect2` — `brew install libfreenect2`.
"""

import collections
import threading
import time

import cv2
import numpy as np

BACKENDS = ("off", "kinect", "kinect2", "server", "sim")

# Where the privileged helper publishes frames. Claiming the Kinect's USB
# interfaces needs root on macOS — the sensor is a composite device and the
# kernel holds the audio interface — but nothing else about this app does, so
# only the helper runs elevated and the app reads frames from it.
SOCKET_PATH = "/tmp/magicmirror-depth.sock"

# Working volume in millimetres. Anything outside is treated as background, so
# these bracket "the person in front of the mirror" rather than the whole room.
DEFAULT_NEAR_MM = 600
DEFAULT_FAR_MM = 3000


class MonoDepth:
    """Depth estimated from the colour camera, so no sensor is needed.

    Depth Anything V2 Small is 25M parameters and runs in about 20ms a frame on
    this hardware. It separates subject from background cleanly even in a dim
    room, which is exactly where edge detection on the colour image gives up.
    """

    REPO = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, device="mps", dtype=None, repo=None):
        import torch

        self.device = device
        self.dtype = dtype or (torch.float16 if device == "mps" else torch.float32)
        self.repo = repo or self.REPO
        self._model = None
        self._processor = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._processor = AutoImageProcessor.from_pretrained(self.repo)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self.repo, torch_dtype=self.dtype).to(self.device).eval()
        self._torch = torch

    def relative(self, frames):
        """RGB frames -> relative depth in 0..1, 1 being nearest.

        The whole batch goes through in one pass; four separate calls cost far
        more than one call with four images.
        """
        self._ensure()
        torch = self._torch
        inputs = self._processor(images=list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype) for k, v in inputs.items()}
        with torch.no_grad():
            predicted = self._model(**inputs).predicted_depth
        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)
        height, width = frames[0].shape[:2]
        predicted = torch.nn.functional.interpolate(
            predicted.float(), size=(height, width), mode="bicubic",
            align_corners=False)[:, 0]
        out = []
        for plane in predicted:
            lo, hi = plane.min(), plane.max()
            out.append(((plane - lo) / (hi - lo + 1e-6)).cpu().numpy())
        return out


def raw_to_mm(raw):
    """Kinect v1 raw 11-bit disparity -> millimetres.

    The usual closed-form fit for this sensor. 2047 is its "no reading" value.
    """
    raw = np.asarray(raw, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        mm = 1000.0 / (raw * -0.0030711016 + 3.3309495161)
    mm[(raw >= 2047) | ~np.isfinite(mm) | (mm < 0)] = 0.0
    return mm


def depth_map_from_mm(depth_mm, near=DEFAULT_NEAR_MM, far=DEFAULT_FAR_MM):
    """Millimetres -> uint8 map, nearest white, outside the working volume black.

    This is where a real depth sensor earns its place over estimated depth: the
    cut-off is metric, so everything behind the person — the wall, the room —
    drops to black and the subject stands out cleanly.
    """
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = (depth > 0) & (depth >= near) & (depth <= far)
    out = np.zeros(depth.shape, dtype=np.uint8)
    # 1..255 for valid pixels, so a genuine far-edge reading is never mistaken
    # for "no reading".
    scaled = 1.0 + (far - depth) / max(1.0, float(far - near)) * 254.0
    out[valid] = np.clip(scaled[valid], 1, 255).astype(np.uint8)
    # Kinect speckle sits right on the silhouette edge.
    return cv2.medianBlur(out, 5)


def sketch_from_depth_map(depth_u8, near=DEFAULT_NEAR_MM, far=DEFAULT_FAR_MM):
    """The contour line drawing, from a map made by depth_map_from_mm."""
    u8 = np.asarray(depth_u8)
    if u8.ndim == 3:
        u8 = u8[:, :, 0]
    mm = np.zeros(u8.shape, dtype=np.float32)
    valid = u8 > 0
    mm[valid] = far - (u8[valid].astype(np.float32) - 1.0) / 254.0 * (far - near)
    return sketch_from_depth(mm, near, far)


def sketch_from_depth(depth_mm, near=DEFAULT_NEAR_MM, far=DEFAULT_FAR_MM,
                      silhouette_weight=1.0, detail_weight=0.8):
    """Depth map in millimetres -> white-on-black line drawing.

    Two things are drawn: the boundary of the working volume, which is the
    subject's outline and by far the strongest line, and the gradient of the
    depth itself, which picks up the folds and features inside that outline.
    """
    depth = np.asarray(depth_mm, dtype=np.float32)
    # 0 means "no reading" on both generations — shadows, too near, too shiny.
    valid = (depth > 0) & (depth >= near) & (depth <= far)
    if not valid.any():
        return np.zeros(depth.shape + (3,), dtype=np.uint8)

    # Normalise within the working volume, with unknowns pushed to the back so
    # they do not read as a surface.
    normalised = np.clip((depth - near) / max(1.0, float(far - near)), 0.0, 1.0)
    normalised[~valid] = 1.0
    # Speckle around depth edges is the usual Kinect artefact.
    normalised = cv2.medianBlur((normalised * 255).astype(np.uint8), 5)

    mask = (valid.astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # A single-pixel outline is too thin to read as a drawn line.
    silhouette = cv2.dilate(cv2.Canny(mask, 50, 150), np.ones((3, 3), np.uint8))

    smooth = cv2.GaussianBlur(normalised, (5, 5), 0).astype(np.float32)
    magnitude = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))
    # Gradients at the edge of the valid region are the mask boundary again,
    # not real structure, so leave those to the silhouette.
    interior = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
    magnitude[~interior] = 0
    # Scale against a high percentile of the interior rather than the maximum:
    # the step from subject to wall dwarfs everything on the subject, and
    # dividing by it flattens the face away to nothing.
    inside = magnitude[interior]
    reference = np.percentile(inside, 97) if inside.size else 0.0
    if reference > 1e-6:
        magnitude = np.clip(magnitude / reference, 0.0, 1.0) * 255.0
    else:
        magnitude[:] = 0

    lines = np.clip(silhouette.astype(np.float32) * silhouette_weight
                    + magnitude * detail_weight, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lines, cv2.COLOR_GRAY2RGB)


def _crop_to_aspect(frame, aspect):
    h, w = frame.shape[:2]
    if w / h > aspect:
        crop_w = int(round(h * aspect))
        x0 = (w - crop_w) // 2
        return frame[:, x0:x0 + crop_w]
    crop_h = int(round(w / aspect))
    y0 = (h - crop_h) // 2
    return frame[y0:y0 + crop_h, :]


class DepthSource:
    """Background thread pulling depth frames and turning them into sketches."""

    def __init__(self, backend, size, stage_aspect, near=DEFAULT_NEAR_MM,
                 far=DEFAULT_FAR_MM, mirror=True, buffer_size=64):
        self.backend = backend
        self.size = size
        self.stage_aspect = stage_aspect
        self.near = near
        self.far = far
        self.mirror = mirror
        self.buffer = collections.deque(maxlen=buffer_size)
        self.latest_raw = None
        self.frames = 0
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread = None
        self._read = None
        self._close = None

    # -- device plumbing -----------------------------------------------------
    def _open_kinect_v1(self):
        import freenect

        # DEPTH_MM needs the registration tables the device sometimes refuses to
        # hand over; DEPTH_11BIT is the raw disparity and needs nothing, so it
        # is tried second and converted here.
        modes = [(freenect.DEPTH_MM, False), (freenect.DEPTH_11BIT, True)]
        chosen = []

        def grab(fmt):
            got = freenect.sync_get_depth(format=fmt)
            return None if got is None else np.asarray(got[0], dtype=np.float32)

        for fmt, raw in modes:
            try:
                frame = grab(fmt)
            except Exception:
                frame = None
            if frame is not None:
                chosen.append((fmt, raw))
                print(f"depth: using {'raw 11-bit disparity' if raw else 'millimetres'}",
                      flush=True)
                break
        if not chosen:
            raise RuntimeError("no depth frame from the Kinect")
        fmt, raw = chosen[0]

        def read():
            frame = grab(fmt)
            return None if frame is None else (raw_to_mm(frame) if raw else frame)

        def close():
            try:
                freenect.sync_stop()
            except Exception:
                pass

        if read() is None:
            raise RuntimeError("no depth frame from the Kinect")
        return read, close

    def _open_kinect_v2(self):
        from pylibfreenect2 import Freenect2, FrameType, SyncMultiFrameListener
        try:
            from pylibfreenect2 import OpenCLPacketPipeline as Pipeline
        except ImportError:
            from pylibfreenect2 import CpuPacketPipeline as Pipeline

        fn = Freenect2()
        if fn.enumerateDevices() == 0:
            raise RuntimeError("no Kinect v2 found")
        device = fn.openDefaultDevice(Pipeline())
        listener = SyncMultiFrameListener(FrameType.Depth)
        device.setIrAndDepthFrameListener(listener)
        device.startStreams(rgb=False, depth=True)

        def read():
            frames = listener.waitForNewFrame()
            depth = np.asarray(frames["depth"].asarray(), dtype=np.float32).copy()
            listener.release(frames)
            return depth

        def close():
            try:
                device.stop(); device.close()
            except Exception:
                pass
        return read, close

    def _open_sim(self):
        import simulate
        return simulate.kinect_reader()

    def try_start(self):
        openers = {"kinect": self._open_kinect_v1, "kinect2": self._open_kinect_v2,
                   "sim": self._open_sim}
        if self.backend not in openers:
            return False
        opener = openers[self.backend]
        try:
            self._read, self._close = opener()
        except ImportError as e:
            print(f"depth disabled — {self.backend} bindings not installed ({e}). "
                  "See the README for the build.", flush=True)
            return False
        except Exception as e:
            print(f"depth disabled — could not open the {self.backend} device ({e})",
                  flush=True)
            return False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"depth: {self.backend} streaming, working volume "
              f"{self.near}-{self.far}mm", flush=True)
        return True

    # -- frames --------------------------------------------------------------
    def _loop(self):
        while not self.stop_flag.is_set():
            try:
                depth = self._read()
            except Exception as e:
                print(f"depth read error: {e}", flush=True)
                time.sleep(0.2)
                continue
            if depth is None:
                time.sleep(0.005)
                continue
            cropped = _crop_to_aspect(depth, self.stage_aspect)
            resized = cv2.resize(cropped, (self.size, self.size),
                                 interpolation=cv2.INTER_NEAREST)
            if self.mirror:
                resized = cv2.flip(resized, 1)
            depth_map = cv2.cvtColor(depth_map_from_mm(resized, self.near, self.far),
                                     cv2.COLOR_GRAY2RGB)
            with self.lock:
                self.buffer.append((time.time(), depth_map))
                self.latest_raw = resized
                self.frames += 1

    def latest_spaced(self, n, span_seconds):
        """Same contract as WebcamCapture, so the worker can treat both alike."""
        with self.lock:
            if len(self.buffer) < n:
                return None
            items = list(self.buffer)
        latest_ts = items[-1][0]
        start_idx = 0
        for i, (ts, _) in enumerate(items):
            if ts >= latest_ts - span_seconds:
                start_idx = i
                break
        recent = items[start_idx:]
        if len(recent) < n:
            recent = items[-n:]
        if n == 1:
            return [recent[-1][1]]
        step = (len(recent) - 1) / (n - 1)
        return [recent[int(round(i * step))][1] for i in range(n)]

    def fresh(self):
        with self.lock:
            return bool(self.buffer) and time.time() - self.buffer[-1][0] < 1.0

    def preview(self):
        """A look at what the model is being given, for the 'h' overlay."""
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1][1].copy()

    def close(self):
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self._close is not None:
            self._close()


class SocketDepthSource:
    """Client for the privileged helper; same interface as DepthSource.

    Reconnects on its own, so either end can be restarted mid-show, and reports
    itself stale when frames stop arriving so the mirror can fall back to the
    camera instead of freezing on the last depth frame.
    """

    # Frames older than this mean the helper has gone quiet.
    STALE_SECONDS = 1.0
    RETRY_SECONDS = 1.0

    def __init__(self, size, path=SOCKET_PATH, buffer_size=64):
        self.size = size
        self.path = path
        self.buffer = collections.deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread = None
        self._sock = None
        self._last_frame = 0.0
        self._connected = False

    def try_start(self):
        """Start the client; succeeds even if the helper is not up yet."""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        deadline = time.time() + 2.0
        while time.time() < deadline and not self._connected:
            time.sleep(0.05)
        if self._connected:
            print(f"depth: connected to the helper at {self.path}", flush=True)
        else:
            print(f"depth: no helper at {self.path} yet — using the camera until "
                  f"one appears.\n  start it with:  sudo .venv/bin/python depth.py "
                  f"--serve", flush=True)
        return True

    def fresh(self):
        return time.time() - self._last_frame < self.STALE_SECONDS

    def _recv_exactly(self, count):
        chunks = []
        while count:
            chunk = self._sock.recv(count)
            if not chunk:
                return None
            chunks.append(chunk)
            count -= len(chunk)
        return b"".join(chunks)

    def _run(self):
        import socket
        import struct

        announced_loss = False
        while not self.stop_flag.is_set():
            try:
                self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._sock.connect(self.path)
            except OSError:
                self._sock = None
                time.sleep(self.RETRY_SECONDS)
                continue
            if announced_loss:
                print("depth: helper back, using depth again", flush=True)
            self._connected = True
            try:
                while not self.stop_flag.is_set():
                    header = self._recv_exactly(4)
                    if header is None:
                        break
                    (length,) = struct.unpack("<I", header)
                    payload = self._recv_exactly(length)
                    if payload is None:
                        break
                    grey = np.frombuffer(payload, dtype=np.uint8).reshape(
                        self.size, self.size)
                    frame = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
                    now = time.time()
                    with self.lock:
                        self.buffer.append((now, frame))
                    self._last_frame = now
            except OSError:
                pass
            finally:
                self._connected = False
                try:
                    self._sock.close()
                except Exception:
                    pass
            if not self.stop_flag.is_set():
                print("depth: lost the helper — falling back to the camera and "
                      "retrying", flush=True)
                announced_loss = True
                with self.lock:
                    self.buffer.clear()
                time.sleep(self.RETRY_SECONDS)

    latest_spaced = DepthSource.latest_spaced
    preview = DepthSource.preview

    def close(self):
        self.stop_flag.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass


def serve(backend="kinect", size=512, stage_aspect=1.0, near=DEFAULT_NEAR_MM,
          far=DEFAULT_FAR_MM, path=SOCKET_PATH, exit_with=None):
    """Own the sensor and publish sketches to any client that connects."""
    import os
    import socket
    import struct

    source = DepthSource(backend, size, stage_aspect, near, far)
    if not source.try_start():
        return 1

    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o777)   # the app runs unprivileged; the helper does not
    server.listen(1)
    server.settimeout(1.0)          # so the parent watch below gets a look in

    def parent_alive():
        # When started by the launcher the helper runs as root and outlives
        # nothing: it goes when the process that started it goes, so quitting
        # the mirror never leaves a root process holding the sensor.
        if exit_with is None:
            return True
        try:
            os.kill(exit_with, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True             # exists, just not ours to signal

    print(f"depth helper: publishing on {path}, working volume {near}-{far}mm "
          f"(ctrl-c to stop)", flush=True)

    try:
        while parent_alive():
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            client.settimeout(None)
            print("depth helper: client connected", flush=True)
            sent = -1
            try:
                while parent_alive():
                    with source.lock:
                        fresh = source.frames != sent and source.buffer
                        frame = source.buffer[-1][1] if fresh else None
                        count = source.frames
                    if frame is None:
                        time.sleep(0.005)      # the sensor runs at 30fps
                        continue
                    payload = frame[:, :, 0].tobytes()
                    client.sendall(struct.pack("<I", len(payload)) + payload)
                    sent = count
            except (BrokenPipeError, ConnectionResetError):
                print("depth helper: client disconnected", flush=True)
            finally:
                client.close()
    except KeyboardInterrupt:
        print("\ndepth helper: stopping", flush=True)
    finally:
        if not parent_alive():
            print("depth helper: the mirror has gone, stopping", flush=True)
        source.close()
        server.close()
        if os.path.exists(path):
            os.unlink(path)
    return 0


def _self_test():
    """`python depth.py` — one clean attempt at the sensor, with a verdict.

    Run this straight after plugging the Kinect in. A failed attempt can leave
    the device's control channel wedged, and the kernel holds the interface
    claimed even after the process dies, so the first try after a replug is the
    one that counts.
    """
    import subprocess

    usb = subprocess.run(["ioreg", "-r", "-c", "IOUSBHostDevice", "-w0"],
                         capture_output=True, text=True).stdout
    for name in ("Xbox NUI Camera", "Xbox NUI Motor", "NuiSensor"):
        if name in usb:
            print(f"usb: {name} present")
    if "Xbox NUI Camera" not in usb and "NuiSensor" not in usb:
        print("usb: no Kinect found — check the cable and the 12V power adapter")
        return 1

    source = DepthSource("kinect", 512, 1.0)
    if not source.try_start():
        print("\nverdict: could not start the sensor.")
        print("  - unplug the Kinect (USB and 12V), wait 5s, plug it back in")
        print("  - connect it straight to the machine, not through a hub or dock")
        print("  - the 12V adapter must be connected; USB alone powers only the motor")
        return 1

    for _ in range(40):
        if source.preview() is not None:
            break
        time.sleep(0.25)
    sketch = source.preview()
    source.close()
    if sketch is None:
        print("\nverdict: opened, but no frames arrived.")
        return 1

    ink = (sketch[:, :, 0] > 40).mean()
    print(f"\nverdict: working. control image is {ink * 100:.1f}% ink "
          f"(5-20% is a good line drawing)")
    out = "/tmp/kinect-check.png"
    cv2.imwrite(out, sketch)
    print(f"wrote {out} — this is exactly what the model would be given")
    if ink < 0.02:
        print("nearly blank: is anyone in the working volume? "
              "try --depth-near/--depth-far")
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true",
                        help="own the sensor and publish frames for the app "
                             "(run this with sudo)")
    parser.add_argument("--backend", choices=("kinect", "kinect2"), default="kinect")
    parser.add_argument("--near", type=int, default=DEFAULT_NEAR_MM)
    parser.add_argument("--far", type=int, default=DEFAULT_FAR_MM)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--exit-with", type=int, default=None, metavar="PID",
                        help="stop when this process exits (used by start.sh)")
    args = parser.parse_args()
    if args.serve:
        sys.exit(serve(args.backend, args.size, 1.0, args.near, args.far,
                       exit_with=args.exit_with))
    sys.exit(_self_test())
