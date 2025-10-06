#!/bin/bash
set -euo pipefail

RTSP_PORT="${RTSP_PORT:-8554}"
ENGINE_DIR="${ENGINE_DIR:-$PWD/models}"

mkdir -p "$ENGINE_DIR"

cmd=(docker run --rm \
  --gpus all \
  --network host \
  -e NVIDIA_DRIVER_CAPABILITIES="video,compute,utility" \
  -e RTSP_PORT="$RTSP_PORT" \
  -e BASE_UDP_PORT="${BASE_UDP_PORT:-5000}" \
  -e PUBLIC_HOST="${PUBLIC_HOST:-127.0.0.1}" \
  -e X264_THREADS="${X264_THREADS:-2}" \
  -v "$ENGINE_DIR":/models \
  batch_streaming:latest)

# Optional pass-throughs
if [[ -n "${SAMPLE_URI:-}" ]]; then cmd+=(-e SAMPLE_URI="$SAMPLE_URI"); fi
if [[ -n "${HW_THRESHOLD:-}" ]]; then cmd+=(-e HW_THRESHOLD="$HW_THRESHOLD"); fi
if [[ -n "${SW_MAX:-}" ]]; then cmd+=(-e SW_MAX="$SW_MAX"); fi
if [[ -n "${MAX_STREAMS:-}" ]]; then cmd+=(-e MAX_STREAMS="$MAX_STREAMS"); fi
if [[ -n "${CTRL_PORT:-}" ]]; then cmd+=(-e CTRL_PORT="$CTRL_PORT"); fi

"${cmd[@]}"
