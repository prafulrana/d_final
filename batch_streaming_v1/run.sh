#!/bin/bash
set -euo pipefail

RTSP_PORT="${RTSP_PORT:-8554}"

docker run --rm \
  --gpus all \
  --network host \
  -e RTSP_PORT="$RTSP_PORT" \
  -e BASE_UDP_PORT="${BASE_UDP_PORT:-5000}" \
  -e PUBLIC_HOST="${PUBLIC_HOST:-10.243.249.215}" \
  batch_streaming:latest
