#!/bin/bash

echo "Building ds-s1 image (--no-cache to ensure fresh build)..."
docker build --no-cache -f Dockerfile.s1 -t ds-s1:latest .

if [ $? -eq 0 ]; then
    echo "✓ ds-s1 image built successfully"
else
    echo "✗ Failed to build ds-s1 image"
    exit 1
fi
