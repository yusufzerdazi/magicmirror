"""Optional super-resolution between the model and the screen.

The diffusion models render at 512px, which a 1080p display then blows up with
a plain resize. This puts a small learned upscaler in between: Real-ESRGAN's
compact video models (SRVGGNetCompact), a few megabytes each and fast enough on
MPS to run on every frame of every batch.

* `lanczos` is a CPU resize — no model, no sharpening, practically free.
* `anime`   is realesr-animevideov3 (16 convs): ~13ms a 16:10 frame on an M5 Pro.
            Made for flat, drawn images, which suits the cartoon prompts.
* `general` is realesr-general-x4v3 (32 convs): ~21ms a 16:10 frame, keeps more
            texture on photographic prompts.

Frames are upscaled 4x and then resized down to the target on the GPU, so
only the finished 1080p frames cross back to the CPU.
"""

import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("off", "lanczos", "anime", "general")
DEFAULT_HEIGHT = 1080

_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
WEIGHTS = {
    # kind: (url, convs in the body)
    "anime": (f"{_RELEASES}/v0.2.5.0/realesr-animevideov3.pth", 16),
    "general": (f"{_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth", 32),
}


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN's compact network: a plain conv stack and a pixel shuffle,
    added to a nearest-neighbour upscale of the input."""

    def __init__(self, num_conv, upscale=4, feat=64):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(3, feat, 3, 1, 1), nn.PReLU(feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(feat, feat, 3, 1, 1), nn.PReLU(feat)]
        body += [nn.Conv2d(feat, 3 * upscale * upscale, 3, 1, 1)]
        self.body = nn.ModuleList(body)
        self.up = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.up(out) + F.interpolate(x, scale_factor=self.upscale,
                                            mode="nearest")


def fetch_weights(kind):
    """Download (once, into torch's hub cache) and return the state dict."""
    url, _ = WEIGHTS[kind]
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu",
                                               progress=False)
    return state.get("params_ema", state.get("params", state))


def output_size(aspect, height=DEFAULT_HEIGHT):
    """(width, height) of a frame `height` tall at `aspect`, width kept even."""
    return int(round(height * aspect / 2.0)) * 2, height


class Upscaler:
    """Upscales batches of RGB frames; networks load on first use."""

    def __init__(self, device="mps", dtype=torch.float16):
        self.device = device
        self.dtype = dtype if device != "cpu" else torch.float32
        self._nets = {}
        self._failed = set()
        self._lock = threading.Lock()

    def _net(self, kind):
        with self._lock:
            net = self._nets.get(kind)
            if net is not None or kind in self._failed:
                return net
            try:
                net = SRVGGNetCompact(WEIGHTS[kind][1])
                net.load_state_dict(fetch_weights(kind), strict=True)
                net = net.to(self.device, self.dtype).eval()
            except Exception as e:
                print(f"upscaler: could not load {kind} ({e}); using lanczos",
                      flush=True)
                self._failed.add(kind)
                return None
            self._nets[kind] = net
            print(f"upscaler: {kind} ready", flush=True)
            return net

    def preload(self, kind):
        if kind in WEIGHTS:
            self._net(kind)

    def __call__(self, rgbs, kind, size):
        """Upscale a list of same-sized uint8 RGB frames to `size` (w, h)."""
        if kind == "off" or not rgbs:
            return rgbs
        net = self._net(kind) if kind in WEIGHTS else None
        if net is None:
            return [cv2.resize(r, size, interpolation=cv2.INTER_LANCZOS4)
                    for r in rgbs]
        width, height = size
        batch = np.ascontiguousarray(np.stack(rgbs))
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device)
            x = x.permute(0, 3, 1, 2).to(self.dtype).div_(255.0)
            y = net(x)
            if y.shape[-2:] != (height, width):
                # Downscaling a 4x image: antialiased, or the fine detail the
                # network just made turns to aliasing. (`area` would be the
                # obvious choice, but MPS only pools by whole factors.)
                y = F.interpolate(y.float(), size=(height, width), mode="bilinear",
                                  antialias=True, align_corners=False)
            y = y.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
            out = y.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return list(out)
