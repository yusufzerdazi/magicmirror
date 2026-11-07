#!/bin/bash
# Start the magic mirror with one command.
#
#   ./start.sh                     # the event setup; uses the Kinect if plugged in
#   ./start.sh --no-kinect         # ignore the Kinect
#   ./start.sh --windowed          # anything else is passed to app.py
#
# With a Kinect attached the mirror runs under sudo, because claiming the sensor
# over USB needs root on macOS, so you will be asked for your password. Without
# one it runs as you. Either way it is a single foreground Python process.
#
# Working volume, in millimetres:  KINECT_NEAR=600 KINECT_FAR=2500 ./start.sh

set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
ME=$(id -un)
NEAR=${KINECT_NEAR:-600}
FAR=${KINECT_FAR:-3000}
EVENT=(--voice --voice-prompt)

use_kinect=1
app_args=()
for arg in "$@"; do
    if [ "$arg" = "--no-kinect" ]; then
        use_kinect=0
    else
        app_args+=("$arg")
    fi
done

if [ "$use_kinect" = 1 ] && ioreg -p IOUSB -w0 -l 2>/dev/null | grep -q '"Xbox NUI Camera"'; then
    echo "kinect found — the mirror needs sudo to reach it"
    sudo -v || { echo "no sudo — run ./start.sh --no-kinect to use the camera"; exit 1; }

    # A root run can leave root-owned files in the model cache, which would then
    # stop a normal run from writing to it. Hand any leftovers back first.
    sudo chown -R "$ME" "$HOME/.cache/huggingface" "$HOME/.cache/torch" __pycache__ 2>/dev/null

    # HOME passed explicitly, so root finds your model cache rather than its own
    # and does not start downloading everything again.
    sudo env HOME="$HOME" "$PY" app.py "${EVENT[@]}" \
        --depth kinect --depth-near "$NEAR" --depth-far "$FAR" \
        ${app_args[@]+"${app_args[@]}"}
    status=$?

    # And again on the way out, if sudo still remembers the password.
    sudo -n chown -R "$ME" "$HOME/.cache/huggingface" "$HOME/.cache/torch" __pycache__ 2>/dev/null
    exit $status
fi

[ "$use_kinect" = 1 ] && echo "no kinect found — using the camera"
exec "$PY" app.py "${EVENT[@]}" ${app_args[@]+"${app_args[@]}"}
