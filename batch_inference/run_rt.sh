#!/bin/bash

# Stop and remove existing container
docker stop ds-s1 2>/dev/null
docker rm ds-s1 2>/dev/null

# Run ds-rt-add container
echo "Starting ds-s1 container..."
docker run -d \
    --name ds-s1 \
    --gpus all \
    --net=host \
    -v /root/d_final/config:/config \
    -v /root/d_final/models:/models \
    ds-rt-add:latest

if [ $? -eq 0 ]; then
    echo "✓ Container started"
    sleep 3
    echo ""
    echo "Container logs:"
    docker logs ds-s1 2>&1 | tail -10
else
    echo "✗ Failed to start container"
    exit 1
fi
