#!/bin/bash
set -e

RELAY_IP="34.14.140.30"

# Clean up any existing containers first
echo "Cleaning up existing containers..."
docker rm -f drishti-s5 2>/dev/null || true

echo "Starting s5 pipeline (pure C++ - zero-copy GPU)..."

# Pipeline 5: PeopleSegNet instance segmentation (pure C++ - zero-copy GPU)
docker run -d --name drishti-s5 --gpus all --rm --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  ds_python:latest \
  /app/deepstream_app \
  rtsp://$RELAY_IP:8554/in_s5 \
  rtsp://$RELAY_IP:8554/s5 \
  /config/pgie_peoplesegnet.txt

echo "Pipeline started:"
echo "  s5: PeopleSegNet segmentation (http://$RELAY_IP:8889/s5/)"
echo ""
echo "View logs: docker logs -f drishti-s5"
