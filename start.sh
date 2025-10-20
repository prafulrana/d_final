#!/bin/bash

# Stop and remove existing containers
docker stop ds-s0 ds-s1 ds-s2 2>/dev/null
docker rm ds-s0 ds-s1 ds-s2 2>/dev/null

# Start s0 (live RTSP input from in_s0 using C binary)
docker run -d \
  --name ds-s0 \
  --gpus all \
  --net=host \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/models":/models \
  --restart unless-stopped \
  ds-s1:latest \
  /app/live_stream 0

# Start s1 (live RTSP input from in_s1 using C binary)
docker run -d \
  --name ds-s1 \
  --gpus all \
  --net=host \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/models":/models \
  --restart unless-stopped \
  ds-s1:latest \
  /app/live_stream 1

# Start s2 (live RTSP input from in_s2 using C binary)
docker run -d \
  --name ds-s2 \
  --gpus all \
  --net=host \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/models":/models \
  --restart unless-stopped \
  ds-s1:latest \
  /app/live_stream 2

# Start publisher (test video to in_s2)
docker run -d \
  --name publisher \
  --net=host \
  publisher:latest

echo "Started ds-s0, ds-s1, ds-s2, and publisher"
docker ps | grep "ds-s\|publisher"
