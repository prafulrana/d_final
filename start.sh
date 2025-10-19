#!/bin/bash

# Stop and remove existing containers
docker stop ds-s0 ds-files 2>/dev/null
docker rm ds-s0 ds-files 2>/dev/null

# Start s0 (live RTSP input from in_s0)
docker run -d \
  --name ds-s0 \
  --gpus all \
  --net=host \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/models":/models \
  --restart unless-stopped \
  nvcr.io/nvidia/deepstream:8.0-triton-multiarch \
  deepstream-app -c /config/s0_live.txt

# Start ds-files (s3/s4 file loops)
docker run -d \
  --name ds-files \
  --gpus all \
  --net=host \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/models":/models \
  --restart unless-stopped \
  nvcr.io/nvidia/deepstream:8.0-triton-multiarch \
  deepstream-app -c /config/file_s3_s4.txt

echo "Started ds-s0 (live RTSP) and ds-files (s3/s4 file loops)"
docker ps | grep "ds-s0\|ds-files"
