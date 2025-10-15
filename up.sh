#!/bin/bash
set -e

RELAY_IP="34.14.140.30"

# Clean up any existing containers first
echo "Cleaning up existing containers..."
docker rm -f drishti-s4 2>/dev/null || true

echo "Starting s4 pipeline (PeopleSemSegNet ONNX Semantic Segmentation)..."

# Pipeline 4: PeopleSemSegNet semantic segmentation (ONNX, C++ with standard nvosd)
docker run -d --name drishti-s4 --gpus all --rm --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  ds_python:latest \
  /app/deepstream_app \
  rtsp://$RELAY_IP:8554/in_s4 \
  rtsp://$RELAY_IP:8554/s4 \
  /config/pgie_peoplesemseg_onnx.txt

echo "Pipeline started:"
echo "  s4: PeopleSemSegNet ONNX Semantic Segmentation (http://$RELAY_IP:8889/s4/)"
echo ""
echo "View logs: docker logs -f drishti-s4"
