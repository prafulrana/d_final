#!/bin/bash

# Stop and remove existing container
docker stop ds-single 2>/dev/null
docker rm ds-single 2>/dev/null

# Run ds-single container
echo "Starting ds-single container..."
docker run -d \
    --name ds-single \
    --gpus all \
    --net=host \
    -v /root/d_final/config:/config \
    -v /root/d_final/models:/models \
    ds-single:latest

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
