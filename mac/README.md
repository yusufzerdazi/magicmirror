# magicmirror — local build

> **London Decompression 2026:** `./start.sh` runs the DREAMWAVE edition. See
> [DREAMWAVE](#dreamwave--london-decompression-2026) below.

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

## DREAMWAVE — London Decompression 2026

`./start.sh` now dresses the mirror for Decompression's 2026 theme, *Dreamwave:
tune in, the dream is real*. It is the default (`--theme dreamwave`);
`--theme mischief` brings back the original cat and coin.

The mirror becomes a radio receiver for dreams, and the screen borrows the
poster: a pink classical archway opening onto the picture, the coral "tune in"
ribbon across its top, a dusk sky of paper clouds and gold constellations
around it, and the glowing coin, now a gold eye on a radio mast.

**Stations.** The dial in the corner is always tuned to a station, and each
station is a dream the people in front of the mirror are redrawn into: Cloud
Walkers, Radio Saints, Moon Ballroom, Deep Sea Static, Playa Mirage, The
Hypnotist, Tin Radio Heart, Garden of Lost Keys (`dreamwave.STATIONS`). Every
50 seconds or so it retunes. The needle sweeps, static rolls over the picture,
and the prompt *morphs* between the two stations rather than cutting, because
the embeddings are interpolated. An empty room channel-surfs faster (26s) to
catch the eye, and **someone new walking up tunes a fresh station**, so each
arrival gets a dream of their own.

**On air.** Speaking into the handset goes live. The dial shows ON AIR with the
words, which become the dream (the "turn me into…" is stripped), and the mirror
answers in the earpiece like a late-night host from an alternate reality.
After 45s it goes back to the stations. Greetings and questions are answered
but not drawn. Anything that would have it draw people naked, sexualised or
hurt is heard and shown, never rendered onto anyone: consent underpins the
event, and the mirror redraws whoever walks past.

**Dream World** (`world.py`) is the first step towards the "where it's going"
part of the art application: a world built around the people in the room
rather than a flat mirror.

* Each station also has a *place*. It is generated, given depth by Depth
  Anything, and drawn as a 2.5D scene with head-coupled parallax. Move sideways
  and the near parts of the world slide past the far ones; step towards the
  mirror and the camera dollies in.
* People are cut out and composited *into* it, with a faint gold rim where
  they meet it. With a Kinect the cut-out is metric: the empty room's depth is
  learned through the night, including the floor that creeps into the working
  volume and any furniture, and anything nearer than it is a person. Kinect IR
  shadows are handled so that a patch of floor is never cut out as a person.
  Without a Kinect, estimated depth does the job.
* What you say still restyles *you*, as in the mirror, so you can become
  something and then walk around inside the world as it.
* `--world auto` (default) alternates mirror and world from one station to the
  next; `on`/`off` hold it. `w` switches it on or off from whatever is showing;
  `world auto|on|off` at the terminal and the "dream world" row in the
  settings panel do the same.

**The Kinect can be switched off live** with `k` (or `kinect on|off`, or the
"kinect" row in the panel). Off, the mirror runs from the camera alone:
estimated depth drives the model and the cut-outs, and presence comes from
motion. Useful if the sensor misbehaves at the venue. With no camera, the
Kinect is the only input and stays on.

World generation runs on the inference thread between batches. The picture
never stalls; a new place arrives a fraction of a second later.

**The frame.** The archway is carved stone, lit from the upper left, casting a
shadow out onto the sky and in onto the picture. The clouds behind it are
photographic cut-outs (`assets/clouds/`) made with the mirror's own model as
white cumulus on black, keyed out and lit pink from below at runtime. Prompts
avoid "poster", "card", "portrait" and "archway", which made the model draw a
second frame inside the arch.

| key | |
| --- | --- |
| `t` | next station (also resumes the dial after a typed prompt) |
| `w` | Dream World on / off |
| `k` | Kinect on / off (off = camera only) |
| `p` | type a prompt; this *holds* the dial on it until `t` |

**Tuning the Kinect at the venue** no longer needs a restart: the settings panel
(`c`) has `kinect near` / `kinect far` rows, 100mm a step. Bring `far` in until
the back wall disappears from the `h` preview.

### Rehearsing without the rig

`simulate.py` scripts a scene: people walk in, dance, lean in towards the
mirror, are joined by a friend and leave, with empty stretches in between. A
simulated camera and Kinect both see it, and the Kinect side has the real
sensor's faults (IR shadows, speckle holes along edges, floor creeping into
the volume).

```sh
./start.sh --simulate                       # full event setup, fake camera + Kinect
.venv/bin/python app.py --camera sim --depth sim --windowed
.venv/bin/python simulate.py /tmp/sheet.png # contact sheet of the script itself
```

For automated runs, `--headless --run-for SECONDS --dump-frames DIR` renders
without a window. Remarks can be piped in as `say …` lines on stdin:

```sh
(sleep 20; echo "say turn us into glowing jellyfish"; sleep 20; echo quit) |
  .venv/bin/python app.py --camera sim --depth sim --headless \
    --dump-frames /tmp/frames --voice-prompt --no-banter
```

### Measured (M5 Pro, simulated input, 1728x1080)

| | warped | display |
| --- | --- | --- |
| sdxs-512 + Kinect, mirror | ~19-20 fps | ~55 fps |
| sdxs-512 + Kinect, Dream World | ~18-19 fps | ~50 fps |
| sdxs-512, camera only (or `k` off) | ~14 fps | ~49 fps |
| sdxl-turbo | ~7 fps | ~49 fps |

A retune or a new world costs a dip of a second or so.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_models.py     # SDXL-Turbo, TAESDXL, Whisper, moderation
```

## Running

The one command, which is what to use at the event:

```sh
./start.sh                       # voice on, Kinect if one is plugged in
./start.sh --no-kinect           # ignore the Kinect
./start.sh --windowed            # any other flags go straight to app.py
KINECT_FAR=2500 ./start.sh       # tighter working volume for a small room
./start.sh --frame 16:10 --upscaler anime  # full-screen 16:10, upscaled to 1080p
```

It is a single foreground Python process. If it finds a Kinect it runs the
mirror under `sudo` — claiming the sensor needs root on macOS — and asks for your
password; without one it runs as you.

Running as root can leave root-owned files in the model cache, which would then
stop a normal run from writing to it, so `start.sh` hands any such files back to
you before starting and again on the way out. It also passes `HOME` through
explicitly, so root uses your model cache instead of downloading everything
again into its own. Run `download_models.py` as yourself beforehand so nothing
needs fetching during a root run at all.

Or directly:

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
| frame | `square` or `16:10` (see [Frame and upscaler](#frame-and-upscaler)) |
| crop position | where the 16:10 band sits in the square, 0 top to 1 bottom |
| upscaler | `off`, `lanczos`, `anime`, `general` |

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
frame 16:10        # or square
upscaler anime     # off / lanczos / anime / general
```

Switching model loads the new pipeline on a background thread and swaps it in
when ready, so the picture keeps running on the old one meanwhile — the same
thing `mischief 3.0`'s `/reload_model` does.

### Keys

| key | |
| --- | --- |
| `q` or `esc` | quit |
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
| `sd-turbo` | 0.87B | **10.2 fps** | SD2.1, natively 512. Implemented but **not offered** — see below |
| `sdxs-512` | 0.32B | **19 fps** | fastest, and tracks properly via its sketch ControlNet — see below |
| `sdxl-lightning` | 2.57B | **3.2 fps** at its 8 steps | `mischief 3.0`'s model. Implemented but **not offered** — see below |

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
untethered. Every frame is CLAHE-equalised before edges are taken. For
ControlNet models `strength` is the `controlnet_conditioning_scale` — how hard
the structure is imposed.

The `edges` row in the config panel chooses how the line drawing is made:

| | | cost |
| --- | --- | --- |
| `kinect` | real depth from a Kinect, background cut by distance — only offered with one attached | ~15 fps |
| `depth` | subject outline from estimated depth, detail from the colour image inside it | 14.1 fps |
| `depth-raw` | the depth map itself, no edge extraction | 14.7 fps |
| `sobel` | gradient edges, equalised | 19.1 fps |

`depth-raw` is worth knowing about: handing the ControlNet the dense grey depth
field is nothing like the line drawings it was trained on, and it works anyway.
It reads the field as a soft shape prior and returns smoother, better
proportioned forms than lines do — at the cost of the facial detail the lines
carry, since depth cannot see a face.

### Styles that are implemented but not offered

`app.DISABLED_EDGE_STYLES` lists the rest. They all still work; adding one back
to `EDGE_STYLES` puts it in the panel.

| | | cost |
| --- | --- | --- |
| `hed` | learned soft boundaries; the best at faces of anything tried | 13.2 fps |
| `pidinet` | learned, crisper than HED | 11.9 fps |
| `lineart` | learned, sparse clean strokes; the model invents more | 11.0 fps |
| `lineart-anime` | learned; renders dark clothing as dithering that leaks into the output | 12 fps |
| `xdog` | stylised lines | 19 fps |
| `canny` | sparsest | 19 fps |

The learned three come from the ControlNet preprocessor family via
`controlnet_aux`, and are what this ControlNet was built to be fed: a gradient
operator finds noise where a face is, and these find the face. `hed` in
particular places eyes and muzzle on the real ones more convincingly than the
depth styles do. They are held back only because they cost about 6 fps against
`sobel` and add a dependency on the annotator weights.

The `depth` styles run [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
(25M parameters, ~20ms a frame) over the camera. Depth separates subject from
background cleanly in a dim room, which is exactly where colour edges give up —
but used alone it draws every fold as a contour ring and reads as topography
rather than a figure. So the outline comes from depth and the detail from the
colour image within it. The whole batch goes through the estimator in one pass;
four separate calls cost 117ms against 67ms batched. If the estimator will not
load it falls back to `sobel` rather than stopping the show.

No sensor is needed for any of this — see the Kinect section for using a real
depth camera instead.

Worth knowing for later: the fastest published real-time camera img2img on
Apple Silicon is [SDXS-512 converted to CoreML](https://arxiv.org/abs/2605.16259),
at 22.7 fps on an M3 Ultra, and that paper finds CoreML conversion to be the
only UNet acceleration that actually helps on Apple Silicon — quantisation does
not, because the work is compute-bound rather than bandwidth-bound. Nothing here
uses CoreML; it is all PyTorch MPS.

## Kinect

A Kinect v1 gives the ControlNet models real depth, which beats estimated depth
on exactly one thing that matters here: the cut-off is **metric**. Everything
outside the working volume — the wall, the rest of the room — drops to black, so
the person in front of the mirror stands out cleanly whatever is behind them.

Claiming the Kinect's USB interfaces needs root on macOS (it is a composite
device and the kernel holds its audio interface). `./start.sh` runs the whole
mirror under `sudo` when a Kinect is plugged in, with `--depth kinect`, which
opens the sensor inside the mirror process.

There is also a split mode, where only a small helper runs as root and the
mirror runs as you, reading depth over a local socket:

```sh
sudo .venv/bin/python depth.py --serve --near 600 --far 3000   # terminal 1
.venv/bin/python app.py --depth server --voice --voice-prompt   # terminal 2
```

It keeps root out of the rest of the app, but it is two commands, and the first
version of `start.sh` that tried to launch both — starting the helper under
`sudo` in the background — could stall silently: a background job cannot read a
password from the terminal, so if `sudo` asked again it simply stopped.

The working volume, in millimetres, is the main thing to tune at a venue: bring
the far edge in until the back wall disappears. With `start.sh` that is
`KINECT_NEAR`/`KINECT_FAR`; directly it is `--depth-near`/`--depth-far`, or
`--near`/`--far` on the helper. `h`
shows the depth map the model is being given, which is the quickest way to frame
the sensor. `sudo .venv/bin/python depth.py` on its own is a one-shot check that
writes `/tmp/kinect-check.png`.

The Kinect is **its own edge style, `kinect`**. It appears in the panel only when
a Kinect is attached, and it is what sdxs starts on when there is one; without a
Kinect sdxs starts on `depth-raw` as before. `depth` and `depth-raw` always mean
depth *estimated* from the colour camera, so each style means one thing whatever
is plugged in. `--edges` picks the starting style explicitly.

If depth frames stop arriving for a second while `kinect` is selected — the
sensor drops out, or the helper in split mode goes away — the mirror switches to
`depth-raw` from the camera rather than freezing on the last frame, and picks the
sensor up again when frames return.

### Getting it running on macOS

* **Use a USB 2.0 port.** On a USB 3 port or through a USB 3 hub the control
  channel corrupts from the first command (`send_cmd: Bad tag`, truncated
  registration reads). On USB 2.0 the stock Homebrew library works unmodified.
* **Run the helper with `sudo`.** Without it the device opens and configures but
  no depth frames ever arrive.
* The 12V adapter must be connected. USB alone powers the motor, which is why
  `Xbox NUI Motor` enumerates even when the camera cannot run.
* A run killed mid-stream (`kill -9`, a crash) can leave a stale reply queued in
  the device, and the next open reads it (`send_cmd: Bad cmd 03 != 16`). Unplug
  and replug to clear it; stopping the helper with ctrl-c shuts it down cleanly.
* The Python bindings are not on PyPI for macOS. Build `wrappers/python` from the
  libfreenect source against Homebrew's library. Cython 3 breaks the shipped
  `setup.py`, which probes a version attribute that no longer exists, so compile
  `freenect.pyx` with a minimal `cythonize` setup instead. Remove the installed
  `.so` before copying a rebuilt one over it: overwriting a library that has
  already been loaded invalidates its cached code signature, and the next
  `import freenect` is SIGKILLed with no message.

## Cameras and microphones

Devices are chosen by kind, best first: **a USB camera, microphone or speaker,
then the machine's built-in one, then a paired iPhone.** Continuity Camera is last on
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

Microphones and speakers need none of this: sounddevice reports input names
alongside its own indices, and `say -a '?'` lists the outputs with the ids `say`
itself wants. Those ids are what get passed — giving `say -a` a name it does not
recognise crashes it rather than falling back.

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

Every line **flows into the cat**. Each takes its own route in from off screen
and ends at the coin, so the mirror reads as listening to the room rather than
as text being scrolled past it. The route is generated from the line itself, so
it is stable for that line's life and no two are alike: the approach direction,
the number of waves and the wobble all vary. That wobble tapers to nothing as
the line nears the coin, so it settles in rather than thrashing about on
arrival.

The curve is a Catmull-Rom spline through its waypoints — building each wave as
its own bezier instead leaves the tangent reversing at every joint, and the text
visibly scallops along a row of half-circles rather than flowing.

Lines are dealt into four lanes, and a lane is a *sector of approach* rather
than a height, so lines converge from different directions instead of stacking.
Alternate lanes take opposite sides, and every approach stays within
`PATH_MAX_ELEVATION` (45 degrees) of horizontal — the coin sits near the bottom
of the frame, so a steeper line has almost no screen to cross and arrives before
it can be read.

The lane comes from a hash of the line rather than from its arrival time. It was
originally quantised from the timestamp into 2.5s buckets, which aliased exactly
onto the four lanes at the ~10s cadence transcripts actually arrive on, so every
line drew the same lane and they all came in from the same side. Glyphs take the path's tangent, and on a line
running right to left they are laid down in reverse and turned through half a
turn, which keeps the text the right way up and still reading forwards.

Each line is tinted white, coin-blue or coin-pink. It fades up as it enters and
dissolves again as it reaches the coin, so text disappears into the cat rather
than sliding under it.

Glyphs are drawn and rotated at three times their final size and then
area-averaged down; rotating at the display size leaves the diagonals ragged.
The synthetic bold stroke is gone too — Sigmar is heavy enough without it, and
the stroke was only thickening the edges.

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

### Answering back

The mirror replies out loud to what it hears. A local language model
(`mlx-community/Llama-3.2-1B-Instruct-4bit`, ~0.1-0.5s a line) writes one short
cheeky sentence and macOS's own `say` speaks it, so nothing leaves the machine
and there is no TTS dependency to install. The reply is spoken only — it is deliberately not put on screen, so the text
belongs to the people in the room. It is composed and spoken on its own thread,
and the transcript reaches the screen and the prompt before the reply is even
queued, so answering back cannot hold either of them up.

```sh
--no-banter                # stay quiet
--banter-chance 0.4        # answer four remarks in ten (default 0.7)
--banter-voice "Bad News"  # any macOS voice; the novelty ones suit the theme
--no-banter-llm            # canned lines only, no model
```

Replies come out of **the handset's earpiece**, not the room: `--speaker`
picks the output the same way `--camera` and `--mic` pick theirs, preferring the
USB device. Because the earpiece is at someone's ear and the microphone is in
the same handset, there is no feedback path and nothing needs muting.

If you ever run it with the speaker open to the room, add `--banter-mute-mic`:
the mirror will otherwise hear its own reply, transcribe it and answer itself,
on and on. That flag mutes the microphone while it talks and discards audio
captured during speech rather than buffering it.

A remark arriving while a reply is already in flight is dropped rather than
queued, so it never builds a backlog to work through.

### The idle prompt

After `--idle-after` seconds of silence (default 40, `0` disables) a centred
call to action eases up over the picture — a headline with one suggestion under
it, taking about three and a half seconds to arrive on a smoothstep rather than
a linear ramp.

An appearance then **runs for 30 seconds and ends**, whether or not anyone has
spoken in the meantime, and cannot come back for **five minutes**. It is a nudge
rather than a nag, and it should not sit there for as long as a room happens to
stay quiet. Each appearance holds one suggestion start to finish; the next takes
the next line in the list. The lines sit still — the only movement is the fade in and out. `overlay.IDLE_VISIBLE_SECONDS` and
`IDLE_REPEAT_SECONDS` set the two intervals. Edit `overlay.IDLE_HEADLINE` and `overlay.IDLE_SUGGESTIONS` for the
wording.

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

The coin and the transcript text follow the stage shape: a square stage gets
`mischief 3.0`'s 9% coin, a wide one a 5.5% coin — about the same size on
screen. Text is the same size in both.

### Frame and upscaler

`--frame 16:10` gives a picture that spans the full width of the screen with no
borders at the sides. The model still renders its 512px square (from a centre
square of the camera or Kinect), and a full-width 16:10 band — 512×320 — is cut
out of what it draws. Nothing is stretched; the cost is the top and bottom of
the square. `--crop-bias` (or *crop position* in the panel) moves the band: `0`
keeps the top of the square, `0.5` (default) the middle, `1` the bottom. 16:10 is
the MacBook display's own shape, so the picture fills it (bar a 2px strip the
menu bar leaves); on a 16:9 TV it would be pillarboxed slightly at the sides.

`--upscaler` takes the cropped frames up to 1080 pixels tall (1728×1080, or
1080×1080 for the square) before they reach the screen:

| `--upscaler` | what | cost per 16:10 frame | generated fps (sdxs, 16:10) |
| --- | --- | --- | --- |
| `off` (default) | the display resize does it | — | ~13.8 |
| `lanczos` | plain CPU resize | ~1ms | ~13 |
| `anime` | Real-ESRGAN `realesr-animevideov3` | ~13ms | ~11.5 |
| `general` | Real-ESRGAN `realesr-general-x4v3` | ~21ms | ~10.5 |

Both Real-ESRGAN models are the compact video networks (2.4MB and 4.7MB),
batched on MPS; `anime` suits the flat cartoon prompts and is the one to pick
by default, `general` keeps more texture on photographic ones. They sharpen
edges and fur noticeably next to a resize. Weights download into
`~/.cache/torch/hub` on first use (`download_models.py` fetches them ahead of
time). The square frame costs roughly 1.6× as much to upscale as the 16:10 band,
being 1.6× the pixels.

Frames are fitted to the stage once as they arrive rather than on every display
tick, so the cross-fade blends screen-sized frames and the display rate stays
around 49fps with the upscaler on.

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
