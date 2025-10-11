#!/bin/bash
set -e

CONTAINER_NAME="drishti"

echo "Building Docker image..."
docker build -t ds_python:latest .

echo "Cleaning up old containers..."
# Kill and remove container by name if it exists
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
# Also kill any orphaned containers using the image
docker ps -q --filter "ancestor=ds_python:latest" | xargs -r docker kill 2>/dev/null || true

echo "Running container in detached mode..."
docker run -d --name "$CONTAINER_NAME" --gpus all --rm --network host -v $PWD/models:/models ds_python:latest

echo "Container started: $CONTAINER_NAME"
echo "View logs with: docker logs -f $CONTAINER_NAME"
