#!/bin/bash
set -euo pipefail

RTSP_PUSH_URL_TMPL="${RTSP_PUSH_URL_TMPL:-rtsp://34.14.144.178:8554/s%u}"
ENGINE_DIR="${ENGINE_DIR:-$PWD/models}"
MAX_STREAMS="${MAX_STREAMS:-2}"
CONTAINER_NAME="${CONTAINER_NAME:-drishti}"

mkdir -p "$ENGINE_DIR"

# Ensure only one sender (drishti) is running: stop and remove all existing containers
echo "Ensuring a single sender is running (stopping old batch_streaming containers)..."
EXISTING_IDS=$(docker ps -a --filter ancestor=batch_streaming:latest --format '{{.ID}}' || true)
if [[ -n "$EXISTING_IDS" ]]; then
  echo "$EXISTING_IDS" | xargs -r docker rm -f >/dev/null || true
fi

# Build docker run command: ensure all -e flags come before the image name
cmd=(docker run -d --rm \
  --gpus all \
  --network host \
  --name "$CONTAINER_NAME" \
  -e NVIDIA_DRIVER_CAPABILITIES="video,compute,utility" \
  -e X264_THREADS="${X264_THREADS:-2}" \
  -e RTSP_PUSH_URL_TMPL="$RTSP_PUSH_URL_TMPL" \
  -e MAX_STREAMS="$MAX_STREAMS" \
  -v "$ENGINE_DIR":/models)

# Optional pass-throughs
if [[ -n "${SAMPLE_URI:-}" ]]; then cmd+=(-e SAMPLE_URI="$SAMPLE_URI"); fi
if [[ -n "${HW_THRESHOLD:-}" ]]; then cmd+=(-e HW_THRESHOLD="$HW_THRESHOLD"); fi
if [[ -n "${SW_MAX:-}" ]]; then cmd+=(-e SW_MAX="$SW_MAX"); fi
if [[ -n "${CTRL_PORT:-}" ]]; then cmd+=(-e CTRL_PORT="$CTRL_PORT"); fi

# Append image name last
cmd+=(batch_streaming:latest)

CID=$("${cmd[@]}")
echo "Launched $CONTAINER_NAME (container: $CID) -> $RTSP_PUSH_URL_TMPL"
