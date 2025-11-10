#!/bin/bash

cd "$(dirname "$0")"

echo "Stopping all containers (1-container machine)..."
docker stop $(docker ps -aq) 2>/dev/null
echo "✓ All containers stopped"

echo ""
echo "Building ds-rt-single container..."
docker build -t ds-rt-single:latest .

if [ $? -eq 0 ]; then
    echo "✓ Build successful"
else
    echo "✗ Build failed"
    exit 1
fi
