"""Camera and microphone selection.

An installation should use whatever good camera is plugged into it, fall back to
the machine's own, and only reach for a paired iPhone if there is nothing else —
picking a Continuity Camera by accident is an easy way to end up filming a phone
on a desk. The same order applies to microphones.
"""

# Best first.
import time

PRIORITY = ("usb", "builtin", "continuity")

_CONTINUITY_HINTS = ("iphone", "ipad", "continuity")
_BUILTIN_HINTS = ("macbook", "built-in", "built in", "internal", "facetime",
                  "imac", "mac mini", "mac studio")


def _classify_by_name(name):
    lowered = name.lower()
    if any(h in lowered for h in _CONTINUITY_HINTS):
        return "continuity"
    if any(h in lowered for h in _BUILTIN_HINTS):
        return "builtin"
    return "usb"


def _camera_dimensions(device):
    """The set of (width, height) a camera can deliver."""
    try:
        import CoreMedia
    except ImportError:
        return set()
    dims = set()
    for fmt in device.formats():
        v = CoreMedia.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
        dims.add((int(v.width), int(v.height)))
    return dims


def resolve_opencv_index(target_index, avf_devices, probe_timeout=0.4):
    """Find which `cv2.VideoCapture` index actually opens `avf_devices[target_index]`.

    OpenCV's AVFoundation backend does not index devices in the order any
    AVFoundation enumeration returns them — on this machine the deprecated API
    and every discovery session list the built-in camera first while OpenCV
    opens a Continuity Camera at index 0. Nothing in the OpenCV API exposes the
    name of the device behind an index, so the mapping is established by
    capability instead: ask each index for a resolution only the wanted camera
    supports, and see which one delivers it.
    """
    import cv2

    if len(avf_devices) < 2:
        return target_index

    target = avf_devices[target_index]
    wanted = _camera_dimensions(target)
    others = set()
    for i, device in enumerate(avf_devices):
        if i != target_index:
            others |= _camera_dimensions(device)
    unique = wanted - others
    if not unique:
        # Indistinguishable by capability (two identical cameras, say); the
        # enumeration order is the only thing left to go on.
        return target_index

    probe_w, probe_h = max(unique)
    opened_any = False
    for index in range(len(avf_devices)):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                continue
            opened_any = True
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, probe_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, probe_h)
            time.sleep(probe_timeout)
            actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            cap.release()
        if actual == (probe_w, probe_h):
            return index
    if not opened_any:
        print("could not open any camera to identify it — check that this "
              "terminal has camera access in System Settings", flush=True)
    return target_index


def list_cameras():
    """Return [(index, name, kind)] as AVFoundation enumerates them.

    These indices are *not* usable with cv2.VideoCapture directly; see
    resolve_opencv_index.
    """
    try:
        import AVFoundation as AV
    except ImportError:
        return []
    out = []
    for index, device in enumerate(AV.AVCaptureDevice.devicesWithMediaType_(
            AV.AVMediaTypeVideo)):
        name = str(device.localizedName())
        kind = None
        is_continuity = getattr(device, "isContinuityCamera", None)
        if is_continuity is not None and is_continuity():
            kind = "continuity"
        else:
            device_type = str(device.deviceType())
            if "BuiltIn" in device_type:
                kind = "builtin"
            elif "External" in device_type:
                kind = "usb"
        if kind is None:
            kind = _classify_by_name(name)
        # A Continuity device can present as External on older systems.
        if kind == "usb" and _classify_by_name(name) == "continuity":
            kind = "continuity"
        out.append((index, name, kind))
    return out


def list_microphones():
    """Return [(index, name, kind)] of input devices, by sounddevice index."""
    try:
        import sounddevice as sd
    except ImportError:
        return []
    out = []
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) < 1:
            continue
        name = device.get("name", f"input {index}")
        out.append((index, name, _classify_by_name(name)))
    return out


def _choose(devices, preference):
    """Pick by explicit index, by kind, or by the standard priority order."""
    if not devices:
        return None, "no devices found"
    if preference not in (None, "auto"):
        text = str(preference)
        if text.isdigit():
            index = int(text)
            for candidate in devices:
                if candidate[0] == index:
                    return candidate, "requested by index"
            return (index, f"index {index}", "unknown"), "requested by index (not enumerated)"
        wanted = "continuity" if text in ("iphone", "ipad") else text
        for candidate in devices:
            if candidate[2] == wanted:
                return candidate, f"requested {wanted}"
        return None, f"no {wanted} device present"
    for kind in PRIORITY:
        for candidate in devices:
            if candidate[2] == kind:
                return candidate, f"{kind} preferred"
    return devices[0], "only device"


def pick_camera(preference=None):
    """(cv2 index, name, kind, why) for the camera to open, or (None, ...)."""
    devices = list_cameras()
    if not devices:
        # Without AVFoundation there is nothing to enumerate; index 0 is the
        # only sensible guess and is what the previous build always used.
        index = int(preference) if str(preference).isdigit() else 0
        return index, f"camera {index}", "unknown", "could not enumerate devices"
    if str(preference).isdigit():
        # An explicit index is taken as an OpenCV index, unprobed.
        index = int(preference)
        name = devices[index][1] if index < len(devices) else f"camera {index}"
        return index, name, "unknown", "requested by index (not probed)"
    chosen, why = _choose(devices, preference)
    if chosen is None:
        return None, None, None, why
    try:
        import AVFoundation as AV
        avf = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
        index = resolve_opencv_index(chosen[0], avf)
    except Exception as e:
        print(f"could not confirm the camera index ({e})", flush=True)
        index = chosen[0]
    if index != chosen[0]:
        why += f", opencv index {index}"
    return index, chosen[1], chosen[2], why


def pick_microphone(preference=None):
    """(index, name, kind, why) for the input device, or (None, ...).

    A None index means "the system default", which is what sounddevice uses.
    """
    devices = list_microphones()
    if not devices:
        return None, "system default", "unknown", "could not enumerate devices"
    chosen, why = _choose(devices, preference)
    if chosen is None:
        return None, None, None, why
    return chosen[0], chosen[1], chosen[2], why


def describe(devices):
    return ", ".join(f"[{i}] {name} ({kind})" for i, name, kind in devices) or "none"
