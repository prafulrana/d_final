#!/bin/bash
set -e

echo "Building batch streaming (DeepStream + direct RTSP push)..."
docker build -t batch_streaming:latest .
echo ""
echo "✓ Build complete!"
echo "Running sanity check inside the image..."
# Run the sanity script inside the freshly built image. This validates required
# GStreamer/DeepStream plugins and that a software H.264 encoder is available.
docker run --rm -i batch_streaming:latest bash -s < sanity.sh
echo "✓ Sanity passed."
echo "Run with: ./run.sh (launches a single \"drishti\" sender to the RTSP target)"
