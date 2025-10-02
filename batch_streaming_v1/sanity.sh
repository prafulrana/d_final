#!/usr/bin/env bash
set -euo pipefail

# Minimal GStreamer/DeepStream sanity check for this project.
# - Run INSIDE the DeepStream container (recommended):
#     bash /opt/nvidia/deepstream/deepstream-8.0/sanity.sh  (if copied there)
#   or from the repo root after building the image:
#     docker run --rm -it --network host batch_streaming:latest bash -lc 'bash -s' < sanity.sh

echo "SANITY: Starting checks"

if ! command -v gst-inspect-1.0 >/dev/null 2>&1; then
  echo "SANITY: gst-inspect-1.0 not found. Run this inside the DeepStream container or pipe via docker run." >&2
  exit 1
fi

echo "SANITY: $(gst-inspect-1.0 --version | head -n1 || true)"

check_elem() {
  local name="$1"
  if gst-inspect-1.0 "$name" >/dev/null 2>&1; then
    echo "OK: $name"
    return 0
  else
    echo "MISS: $name"
    return 1
  fi
}

# Core elements for this app
miss=0
check_elem rtph264pay || miss=1
check_elem udpsrc      || miss=1
check_elem udpsink     || miss=1
check_elem nvvideoconvert || miss=1
check_elem nvosdbin       || miss=1
check_elem nvstreamdemux  || miss=1
check_elem videorate      || miss=1

# Software H.264 encoder: try x264enc, then avenc_h264, then openh264enc
encoder=""
if check_elem x264enc; then encoder="x264enc"; fi
if [[ -z "$encoder" ]] && check_elem avenc_h264; then encoder="avenc_h264"; fi
if [[ -z "$encoder" ]] && check_elem openh264enc; then encoder="openh264enc"; fi

if [[ -z "$encoder" ]]; then
  echo "FAIL: No H.264 software encoder plugin (x264enc/avenc_h264/openh264enc) is available." >&2
  echo "Hint: install gst-plugins-ugly + gst-libav and their runtime libs (libvpx9, libmp3lame0, libflac12t64, libmpg123-0, libdvdnav4, libdvdread8, libdca0, mjpegtools)." >&2
  exit 2
fi

if [[ "$miss" -ne 0 ]]; then
  echo "FAIL: One or more required elements are missing (see MISS above)." >&2
  exit 3
fi

# Quick dry-run encode path (single buffer). Quiet output with -q.
echo "SANITY: Testing $encoder encode path (single buffer)" 
gst-launch-1.0 -q \
  videotestsrc num-buffers=1 ! \
  videoconvert ! videorate ! video/x-raw,framerate=30/1 ! \
  "$encoder" tune=zerolatency speed-preset=ultrafast bitrate=3000 key-int-max=60 bframes=0 ! \
  h264parse ! \
  rtph264pay pt=96 ! \
  fakesink 

echo "PASS: Environment is sane. Using encoder: $encoder"
