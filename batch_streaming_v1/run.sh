#!/bin/bash
set -euo pipefail

RTSP_PORT="${RTSP_PORT:-8554}"
ENGINE_DIR="${ENGINE_DIR:-$PWD/models}"

mkdir -p "$ENGINE_DIR"

docker run --rm \
  --gpus all \
  --network host \
  -e RTSP_PORT="$RTSP_PORT" \
  -e BASE_UDP_PORT="${BASE_UDP_PORT:-5000}" \
  -e PUBLIC_HOST="${PUBLIC_HOST:-10.243.249.215}" \
  -v "$ENGINE_DIR":/models \
  batch_streaming:latest
