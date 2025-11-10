#!/bin/bash

# Stop and remove existing container
docker stop ds-single 2>/dev/null
docker rm ds-single 2>/dev/null

# Run ds-rt-single container (override CMD to use singular Python script)
echo "Starting ds-single container..."
docker run -d \
    --name ds-single \
    --gpus all \
    --net=host \
    -v /root/d_final/config:/config \
    -v /root/d_final/models:/models \
    ds-rt-single:latest \
    python3 -u /app/deepstream_rt_src_add_del_singular.py rtsp://34.47.221.242:8554/in_s0 rtsp://34.47.221.242:8554/in_s1 rtsp://34.47.221.242:8554/in_s2

if [ $? -eq 0 ]; then
    echo "✓ Container started"
    sleep 3
    echo ""
    echo "Container logs:"
    docker logs ds-single 2>&1 | tail -20
else
    echo "✗ Failed to start container"
    exit 1
fi
