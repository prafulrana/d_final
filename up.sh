#!/bin/bash
set -e

RELAY_IP="34.14.140.30"

# Clean up any existing containers first
echo "Cleaning up existing containers..."
docker rm -f drishti-s4 drishti-s5 2>/dev/null || true

echo "Starting s4 pipeline (YOLOv8 COCO Object Detection)..."

# Pipeline 4: YOLOv8 COCO detection (80 classes including person)
docker run -d --name drishti-s4 --gpus all --rm --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/libnvdsinfer_custom_impl_Yolo.so":/app/libnvdsinfer_custom_impl_Yolo.so \
  ds_python:latest \
  /app/deepstream_app \
  rtsp://$RELAY_IP:8554/in_s4 \
  rtsp://$RELAY_IP:8554/s4 \
  /config/pgie_yolov8_coco.txt

echo "Pipeline started:"
echo "  s4: YOLOv8 COCO Detection (http://$RELAY_IP:8889/s4/)"
echo ""
echo "View logs: docker logs -f drishti-s4"
