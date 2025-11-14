#!/bin/bash

echo "Stopping ds-single container..."
docker stop ds-single 2>/dev/null
docker rm ds-single 2>/dev/null

echo "Building ds-single image..."
docker build -t ds-single:latest .

if [ $? -eq 0 ]; then
    echo "✓ ds-single image built successfully"
else
    echo "✗ Failed to build ds-single image"
    exit 1
fi
