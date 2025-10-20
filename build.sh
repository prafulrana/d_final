#!/bin/bash

echo "Building ds-s1 image..."
docker build -f Dockerfile.s1 -t ds-s1:latest .

if [ $? -eq 0 ]; then
    echo "✓ ds-s1 image built successfully"
else
    echo "✗ Failed to build ds-s1 image"
    exit 1
fi

echo "Building publisher image..."
docker build -f publisher/Dockerfile -t publisher:latest publisher/

if [ $? -eq 0 ]; then
    echo "✓ publisher image built successfully"
else
    echo "✗ Failed to build publisher image"
    exit 1
fi
