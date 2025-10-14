#!/bin/bash
set -e

RELAY_IP="34.14.140.30"

# Clean up any existing containers first
echo "Cleaning up existing containers..."
docker rm -f drishti-s0 drishti-s1 drishti-s2 2>/dev/null || true

echo "Starting 3 inference pipelines..."

# Pipeline 0: TrafficCamNet detection (default)
docker run -d --name drishti-s0 --gpus all --rm --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/app.py":/app/app.py \
  -v "$(pwd)/probe_default.py":/app/probe_default.py \
  -v "$(pwd)/probe_yoloworld.py":/app/probe_yoloworld.py \
  -v "$(pwd)/probe_segmentation.py":/app/probe_segmentation.py \
  ds_python:latest \
  python3 -u /app/app.py \
  -i rtsp://$RELAY_IP:8554/in_s0 \
  -o rtsp://$RELAY_IP:8554/s0 \
  -c /config/pgie_resnet_traffic.txt

# Pipeline 1: YOLO-World detection
docker run -d --name drishti-s1 --gpus all --rm --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/app.py":/app/app.py \
  -v "$(pwd)/probe_default.py":/app/probe_default.py \
  -v "$(pwd)/probe_yoloworld.py":/app/probe_yoloworld.py \
  -v "$(pwd)/probe_segmentation.py":/app/probe_segmentation.py \
  ds_python:latest \
  python3 -u /app/app.py \
  -i rtsp://$RELAY_IP:8554/in_s1 \
  -o rtsp://$RELAY_IP:8554/s1 \
  -c /config/pgie_yoloworld.txt \
  --probe probe_yoloworld

# Pipeline 2: PeopleSegNet instance segmentation
docker run -d --name drishti-s2 --gpus all --rm --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/app.py":/app/app.py \
  -v "$(pwd)/probe_default.py":/app/probe_default.py \
  -v "$(pwd)/probe_yoloworld.py":/app/probe_yoloworld.py \
  -v "$(pwd)/probe_segmentation.py":/app/probe_segmentation.py \
  ds_python:latest \
  python3 -u /app/app.py \
  -i rtsp://$RELAY_IP:8554/in_s2 \
  -o rtsp://$RELAY_IP:8554/s2 \
  -c /config/pgie_peoplesegnet.txt \
  --probe probe_segmentation

echo "3 pipelines started:"
echo "  s0: TrafficCamNet detection (http://$RELAY_IP:8889/s0/)"
echo "  s1: YOLO-World detection (http://$RELAY_IP:8889/s1/)"
echo "  s2: PeopleSegNet segmentation (http://$RELAY_IP:8889/s2/)"
echo ""
echo "View logs: docker logs -f drishti-s0"
echo "Monitor all: docker ps --filter ancestor=ds_python:latest"
