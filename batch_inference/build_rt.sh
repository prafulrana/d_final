#!/bin/bash

cd "$(dirname "$0")"

echo "Building ds-rt-add container..."
docker build -t ds-rt-add:latest .

if [ $? -eq 0 ]; then
    echo "✓ Build successful"
else
    echo "✗ Build failed"
    exit 1
fi
