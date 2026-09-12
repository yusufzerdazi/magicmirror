# magicmirror — local build

A single-process Apple Silicon build of the installation. It does locally what
the hosted pair does over the network:

Two frontend revisions are referenced. `local-run` carries the 16:9 build; the
`mischief 3.0` branch carries the square one, with the animated background and
retuned overlay geometry. The local build follows the square one by default.

| hosted | local |
| --- | --- |
| `web/src/components/WarpPage.tsx` — camera capture, stage, cross-fade, coin, transcript text | `app.py` + `overlay.py` |
| `mischief 3.0`'s `PageContainer` — square stage, drifting blurred circles | `overlay.MorphingCircles` |
| `backend/diffusion_processor.py` — SDXL-Turbo img2img, mirror, tint | `app.py` (`InferenceWorker`, `TintCycle`) |
| `backend/settings_api.py` — `/prompt/`, `/secondprompt/`, `/blend/` | `prompting.py` + stdin commands |
| `backend/speech_processor.py` — Whisper, confidence and repetition filters | `voice.py` |
| `backend/safety_checker.py` — transcript moderation | `moderation.py` |
| `web/public/mischief.svg` | `svg_logo.py` + `assets/mischief.svg` |

Constants copied from either side are commented with where they came from, so
the two can be kept in step.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_models.py     # SDXL-Turbo, TAESDXL, Whisper, moderation
```

## Running

```sh
.venv/bin/python app.py                 # matches the hosted defaults
.venv/bin/python app.py --voice         # + microphone, transcription, on-screen text
.venv/bin/python app.py --windowed --voice --voice-prompt
```

### Config panel

`c` opens a settings panel, as `mischief 3.0` does. Arrow keys (or `j`/`k` and
`h`/`l`) move and change, `c` or escape closes:

| | |
| --- | --- |
| model | `sdxl-turbo`, `sd-turbo`, `sdxs-512`, `sdxl-lightning` — press return to load |
| edges | line drawing fed to the ControlNet models: `sobel`, `xdog`, `canny` |
| steps, strength | per-model defaults; changing the model resets them |
| seed, blend | as the keys do |
| cross-fade | `auto` or a fixed step |
| framing, rotation, tint, background | as the flags do |

Arrow keys are not delivered reliably by every HighGUI build, so `j`/`k` move
and `h`/`l` change as well (`w`/`s` and `a`/`d` also work). If a key does arrive
but is not one the panel knows, the footer prints its code rather than ignoring
it silently.

Everything in the panel is also reachable from the terminal, which does not
depend on the window having keyboard focus at all:

```
model              # list the models and show the current one
model sd-turbo     # switch
steps 3
strength 0.5
```

Switching model loads the new pipeline on a background thread and swaps it in
when ready, so the picture keeps running on the old one meanwhile — the same
thing `mischief 3.0`'s `/reload_model` does.

### Keys

| key | |
| --- | --- |
| `q` | quit |
| `f` | toggle fullscreen |
| `h` | show the camera preview and status (the frontend's hidden `H` panel) |
| `r` | rotate 90° |
| `s` | next seed |
| `c` | settings panel |
| `p` | type a new prompt, on screen — enter commits, escape cancels |
| `o` | same, for the second prompt |
| `[` `]` | blend towards the first / second prompt |

In fullscreen the window holds keyboard focus, so `p` is the way to change the
prompt while the installation is running — anything typed at the terminal goes
to the terminal, which is behind the window.

### Terminal commands

The hosted build changes prompts over HTTP. Locally the same things can be
typed at the terminal running the app, which is handy for scripting a launch or
driving it over SSH (it never blocks the render loop — but see the note above
about window focus):

```
a cathedral made of coral        # sets the prompt
prompt a cathedral made of coral
second A psychedelic landscape.
blend 0.4
seed 7
say the mirror is listening      # inject a transcript, as if it had been heard
model sd-turbo                   # switch model (also steps N / strength X)
quit
```

`say` puts text through exactly the path a real transcription takes, so the
scrolling text (and `--voice-prompt`) can be tested without a microphone.

`blend` interpolates the two prompt embeddings, the same thing the backend gets
out of compel's `("a", "b").blend(0.60, 0.40)`.

## Models

`--model`, and the config panel, choose between:

| | UNet | batch of 4 at 512 | notes |
| --- | --- | --- | --- |
| `sdxl-turbo` (default) | 2.57B | **8.0 fps** | what the `local-run` backend uses; best looking of the three |
| `sd-turbo` | 0.87B | **10.2 fps** | SD2.1, natively 512, a third the size; slightly flatter output |
| `sdxs-512` | 0.32B | **19 fps** | fastest, and tracks properly via its sketch ControlNet — see below |
| `sdxl-lightning` | 2.57B | **3.2 fps** at its 8 steps | `mischief 3.0`'s model; needs SDXL base (~7GB) on top of the Lightning UNet |

Measured on this machine at the app's own settings. Lightning is a real
quality/frame-rate trade rather than a free win: 8 steps costs 8.0 fps -> 3.2.

### sdxs-512, and how it became the fastest mirror here

`IDKiro/sdxs-512-dreamshaper` is a 0.32B one-step distilled SD1.5. Pointed at
the ordinary img2img path it produced grey mush, and the reason is structural:

**its UNet is only valid at one timestep.** Plain text-to-image collapses the
moment a second step is taken, with no img2img involved at all:

| txt2img steps | timesteps | output |
| --- | --- | --- |
| 1 | `[999]` | clean, std 63.5 |
| 2 | `[999, 499]` | mush, std 18.6 |
| 4 | `[999, 749, 499, 249]` | worse, std 13.7 |

Distillation maps noise to an image in one shot and leaves nothing usable below
that top timestep. diffusers' img2img has to *start* below it — that is how it
keeps any of the input — so the UNet is always asked for a prediction it never
learned, and no combination of steps and strength avoids that. It is not a
misconfiguration and it is not tunable.

**The model conditions on images through a ControlNet instead**, which is how
the project's own `demo_sketch.py` uses it: one step, guidance 0, and the image
supplied as a *line drawing* to `IDKiro/sdxs-512-dreamshaper-sketch` rather than
as a partly-noised init. That is a text-to-image pass with structural
conditioning, so it only ever needs the one timestep the model has.

So `warp_batch_control` turns each camera frame into a white-on-black line
drawing and runs exactly that. The result tracks pose, framing and scale
properly — and it is the fastest model here:

| | batch of 4 at 512 | conditioning |
| --- | --- | --- |
| `sdxs-512` | **19 fps** | sketch ControlNet |
| `sd-turbo` | 10.2 fps | img2img |
| `sdxl-turbo` | 8.3 fps | img2img |

The camera is usually dim and low contrast, and that matters more than the edge
operator: without equalising first the line drawing comes out nearly blank and
the model free-runs, which was the other half of why early attempts looked
untethered. Every frame is CLAHE-equalised before edges are taken. The `edges`
row in the config panel switches between `sobel` (default, densest and the most
faithful), `xdog` (cleaner, more stylised lines) and `canny` (sparsest, lets the
model invent more). For ControlNet models `strength` is the
`controlnet_conditioning_scale` — how hard the structure is imposed.

Worth knowing for later: the fastest published real-time camera img2img on
Apple Silicon is [SDXS-512 converted to CoreML](https://arxiv.org/abs/2605.16259),
at 22.7 fps on an M3 Ultra, and that paper finds CoreML conversion to be the
only UNet acceleration that actually helps on Apple Silicon — quantisation does
not, because the work is compute-bound rather than bandwidth-bound. Nothing here
uses CoreML; it is all PyTorch MPS.

## Cameras and microphones

Devices are chosen by kind, best first: **a USB camera or microphone, then the
machine's built-in one, then a paired iPhone.** Continuity Camera is last on
purpose — an installation that silently starts filming a phone on a desk is a
bad failure mode. Cameras are classified through AVFoundation
(`isContinuityCamera`, plus the built-in/external device type).

### Why the camera index needs probing

OpenCV's AVFoundation backend does not index cameras in the order any
AVFoundation enumeration returns them, and nothing in the OpenCV API reports the
name of the device behind an index. On this machine the deprecated
`devicesWithMediaType:` API, and every `AVCaptureDeviceDiscoverySession`
regardless of the device-type order asked for, all list the built-in camera
first — while `cv2.VideoCapture(0)` opens the Continuity Camera.

So the mapping is established by capability instead. Each camera's supported
formats are read from AVFoundation, a resolution unique to the wanted camera is
picked, and each index is asked for it in turn; the one that delivers it is the
right index. Here the built-in camera offers 1552x1552 and 1760x1328 (Center
Stage and Desk View) and the iPhone offers 1920x1440, so they separate cleanly:

```
index 0: asked 1552x1552 -> got 1920x1440   (iPhone)
index 1: asked 1552x1552 -> got 1552x1552   (built-in)
```

If two cameras are genuinely indistinguishable by format, it falls back to the
enumeration order. Probing costs a second or two at startup and briefly opens
each camera in turn, which will wake a paired iPhone even when it is not the one
being chosen. `--camera <index>` skips probing entirely.

Microphones need none of this: sounddevice reports names alongside its own
indices.

```sh
.venv/bin/python app.py --list-devices    # what was found, and how each was classified
.venv/bin/python app.py --camera usb --mic builtin
.venv/bin/python app.py --camera 1        # or an explicit index
```

Selection happens at startup; plugging a USB camera in later needs a restart.

## Voice

`--voice` records 10-second chunks (the frontend's `AUDIO_INTERVAL`), transcribes
them, keeps only Whisper segments with `avg_logprob > -1`, drops repetitive
output, runs the text through moderation, and puts what survives on screen for 5
seconds (`SCROLL_DURATION`), scrolling along the same two wavy paths.

The text is drawn over the whole screen rather than inside the picture, so a
transcript sweeps in across the background before it crosses the square. Both
frontends clip it to their content box; this does not. The coin stays anchored
to the picture and above the text, as it is in the frontend.

Widening the sweep must not enlarge the text with it, so the two scales are
separate: the paths are stretched horizontally across the full display, while
the glyph size, the wave height and the vertical position all come from the
stage — the bottom 40% of it, as in the frontend. On a 1084pt square stage that
reproduces `mischief 3.0`'s geometry exactly (33px glyphs on a baseline at
y=976, 65px of wave) no matter how wide the display is.

### Which model

The hosted backend runs Whisper `base`, which falls apart as soon as there is
room noise. Measured here on five prompt-like phrases, clean and mixed with
noise at 10/5/0dB SNR (mean word error rate over all four conditions):

| | WER | latency per 10s chunk | costs the diffusion loop |
| --- | --- | --- | --- |
| **mlx-whisper `large-v3-turbo`, GPU (default)** | **2.8%** | **0.38s** | 2% |
| faster-whisper `distil-large-v3`, CPU | 10.6% | 3.1s | 0% |
| faster-whisper `small`, CPU | 15.3% | 1.5s | 0% |
| faster-whisper `base`, CPU (the hosted model) | 24.7% | 0.5s | 0% |

`large-v3-turbo` is both the most accurate and the fastest of these, and only
runs for a third of a second once per chunk, so sharing the GPU with the
diffusion loop costs about 2% of the frame rate — 8.00 fps against 7.82.

`--whisper-backend faster` switches to the CPU stack (`--whisper-model base`
restores the hosted build exactly); that path keeps beam search, which MLX has
no decoder for.

### Silence

Whisper invents confident text out of room tone — `base` produces "I'm sorry,
I'm sorry, I'm sorry, ..." and `large-v3-turbo` produces "Thank you." The hosted
build only catches this when the invention happens to be repetitive.

Every chunk is therefore gated on Silero VAD before it reaches a transcriber, so
silence is never transcribed at all. It costs ~20ms and it covers both backends.
`--no-speech-gate` turns it off.

Moderation runs by default, as it does on `/transcribe`. The classifier
(`facebook/roberta-hate-speech-dynabench-r4-target`) only catches targeted hate
speech, so `--banned-words` additionally applies the word list that sits unused
in `backend/settings_api.py`.

## Display

The stage is the largest 16:9 rectangle that fits the display, letterboxed in
black, with the warped frame scaled to fill it — the frontend's `PageContainer`
plus its `drawImage` cover maths.

Getting the canvas size exactly right matters more than it looks. HighGUI scales
whatever image it is given to fit the window *while preserving that image's
aspect ratio*, so a canvas even slightly the wrong shape gets letterboxed by the
window in its own grey. A fullscreen HighGUI window is also placed below the
menu bar, so the area it can paint is the display minus the menu bar — that is
what `detect_screen_size()` returns, via `NSScreen.visibleFrame`.
`--display-size` overrides it.

The menu bar strip itself (33pt here) cannot be reclaimed: the window will not
extend under it, and setting `NSApplicationPresentationHideMenuBar` does not
move it. It shows as black above the picture.

`--stage-aspect` changes the shape of the picture area: `16:9` (default) matches
the hosted build, `fill` uses the whole screen with no bars at all, and an
explicit ratio like `16:10` is accepted too. With `--capture-fit wide` the camera
crop follows whatever is chosen, so `fill` gives a full-screen picture with
correct proportions.

### Shape and background

The stage is square by default, as in `mischief 3.0` (`width: min(100vh, 100vw)`),
centred on the display. `--stage-aspect` takes `16:9` for the older layout,
`fill` for the whole screen, or any `W:H`.

Around it sits that build's animated background: a 135-degree gradient through
`#1a8cd8 / #22A1EC / #FFBAF1 / #e89dd9`, with ten circles in the coin's blue and
pink drifting across it under an 80px blur, bouncing off the edges and pulsing
between 20% and 60% of the viewport height on a 50ms tick. `--background black`
turns it off.

It is simulated and drawn at an eighth of the screen size and scaled up — an
80px blur removes everything finer than that anyway — and cached between
simulation ticks, so it costs about 3ms twenty times a second rather than on
every display frame.

The coin and the transcript paths follow the stage shape, as they do in the
frontend: a square stage gets `mischief 3.0`'s 9% coin and paths at y=300 with a
60 unit wave, a wide one gets the 16:9 build's 5% coin and paths at y=350/50.

### Framing — `--capture-fit`

The model only ever sees 512×512, so the choice is which part of the camera it
gets and how it is squeezed in:

| mode | field of view | proportions |
| --- | --- | --- |
| `wide` (default) | the whole camera view | correct |
| `stretch` | cropped to the middle ~56% vertically | stretched 1.8× vertically |
| `crop` | centre square, then cropped again by the stage | correct |

`stretch` is what the frontend does — it squeezes the camera frame into the
capture canvas and the cover-fit then crops it — so it is there for parity, not
because it looks good. `wide` crops the camera to 16:9, squeezes *that* into the
square for the model, and un-squeezes it across the stage: nothing is thrown
away and nobody is stretched.

### Frame rate — `--fade-step`

Frames arrive one inference batch at a time — four at once, then a pause — so
the queue has to be used as a jitter buffer and played out at the rate frames
are *produced*. Draining it as fast as possible is what makes a batch flash past
and then leaves the picture frozen until the next one lands.

`auto` (the default) sets the fade length from the measured arrival rate and
holds about half a batch in hand, nudging gently when the queue drifts.
Measured against the real pipeline at ~8fps:

| `--fade-step` | displayed | gap between frames | lag |
| --- | --- | --- | --- |
| `auto` | 7.9 fps | 123ms ± **17ms** | 370ms |
| `1.0` (hard cut) | 8.2 fps | 121ms ± **176ms** (17–444ms) | 31ms |
| `0.05` (hosted) | 2.9 fps | 350ms ± 4ms | 1950ms |

The hosted build's fixed step is perfectly even but throws away two frames in
three; a hard cut keeps every frame but stutters batch by batch. `auto` costs
370ms of buffering for an even picture at the full rate. The running app prints
`queue` and `dropped` alongside its frame rates.

### Debugging

`--dump-frames DIR` writes the composited frames to disk and `--debug-geometry`
logs the canvas and stage size against the window — between them they show
exactly what reached the screen without needing a screen recorder.

## Where this deliberately differs from the hosted build

Three defaults depart from parity because the hosted behaviour is a visible
regression on local hardware; each is one flag away.

* `--capture-fit wide` instead of the frontend's `stretch`.
* `--fade-step auto` instead of either frontend's fixed `0.05`.
* Whisper `large-v3-turbo` instead of `base`, with silent chunks gated out.

```sh
--capture-fit stretch --fade-step 0.05 \
  --whisper-backend faster --whisper-model base --no-speech-gate
```

restores the hosted behaviour exactly.

## Not carried over

* **Real-ESRGAN 2× upscaling.** The backend upscales every 512px frame to 1080p
  before sending it. Locally the frame is scaled straight to the stage; running
  Real-ESRGAN on MPS would cost more than the diffusion itself.
* **WebSocket transport, password auth, reconnect and health checks.** There is
  no network hop to protect or recover.
* **The OSC controller and the rest of the settings API.** Everything they reach
  is available as a flag or a terminal command.
* **`mischief 3.0`'s DVD mode** (the logo bouncing around the screen every 20
  minutes, with a corner-hit effect) and its turbo/lightning model switch.
  Neither was asked for; both are straightforward to add.
