#!/bin/bash
set -e

RELAY_IP="34.14.140.30"

# Clean up any existing containers first
echo "Cleaning up existing containers..."
docker rm -f drishti-s0 drishti-s1 drishti-s2 2>/dev/null || true

echo "Starting 3 YOLO detection pipelines (s0, s1, s2)..."

# Pipeline s0: YOLOv8 COCO detection
docker run -d --name drishti-s0 --gpus all --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/libnvdsinfer_custom_impl_Yolo.so":/app/libnvdsinfer_custom_impl_Yolo.so \
  ds_python:latest \
  /app/deepstream_app \
  rtsp://$RELAY_IP:8554/in_s0 \
  rtsp://$RELAY_IP:8554/s0 \
  /config/pipeline.txt

# Pipeline s1: YOLOv8 COCO detection
docker run -d --name drishti-s1 --gpus all --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/libnvdsinfer_custom_impl_Yolo.so":/app/libnvdsinfer_custom_impl_Yolo.so \
  ds_python:latest \
  /app/deepstream_app \
  rtsp://$RELAY_IP:8554/in_s1 \
  rtsp://$RELAY_IP:8554/s1 \
  /config/pipeline.txt

# Pipeline s2: YOLOv8 COCO detection
docker run -d --name drishti-s2 --gpus all --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/libnvdsinfer_custom_impl_Yolo.so":/app/libnvdsinfer_custom_impl_Yolo.so \
  ds_python:latest \
  /app/deepstream_app \
  rtsp://$RELAY_IP:8554/in_s2 \
  rtsp://$RELAY_IP:8554/s2 \
  /config/pipeline.txt

echo "Pipelines started:"
echo "  s0: YOLOv8 COCO Detection (http://$RELAY_IP:8889/s0/)"
echo "  s1: YOLOv8 COCO Detection (http://$RELAY_IP:8889/s1/)"
echo "  s2: YOLOv8 COCO Detection (http://$RELAY_IP:8889/s2/)"
echo ""
echo "View logs:"
echo "  docker logs -f drishti-s0"
echo "  docker logs -f drishti-s1"
echo "  docker logs -f drishti-s2"
