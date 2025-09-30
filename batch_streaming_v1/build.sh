#!/bin/bash
set -e

echo "Building batch streaming (4 streams, batch inference)..."
docker build -t batch_streaming:latest .
echo ""
echo "✓ Build complete!"
echo "Run with: ./run.sh"