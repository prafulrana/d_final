#!/bin/bash
set -euo pipefail

STREAMS="${STREAMS:-2}"
RTSP_PORT="${RTSP_PORT:-8554}"

docker run --rm \
  --gpus all \
  --network host \
  -e STREAMS="$STREAMS" \
  -e RTSP_PORT="$RTSP_PORT" \
  -e BASE_UDP_PORT="${BASE_UDP_PORT:-5000}" \
  batch_streaming:latest
