#!/bin/bash
set -e

echo "Building batch streaming (C RTSP server, config-driven pipeline)..."
docker build -t batch_streaming:latest .
echo ""
echo "✓ Build complete!"
echo "Run with: ./run.sh"
