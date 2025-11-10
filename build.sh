#!/bin/bash

echo "Stopping all containers..."
docker ps -qa | xargs -r docker stop 2>/dev/null

echo "Building ds-single image..."
docker build -f Dockerfile -t ds-single:latest .

if [ $? -eq 0 ]; then
    echo "✓ ds-single image built successfully"
else
    echo "✗ Failed to build ds-single image"
    exit 1
fi
