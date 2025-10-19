#!/bin/bash
# Pre-build TensorRT engine for a given inference config
# This avoids rebuilding engines on every container restart
# Usage: ./cache_engine.sh <config_name>

set -e

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <config_name>"
    echo ""
    echo "Available configs:"
    echo "  config_infer_primary.txt    - TrafficCamNet"
    echo "  config_infer_yolov8.txt     - YOLOv8n COCO"
    echo "  config_infer_yoloworld.txt  - YOLOWorld custom"
    echo ""
    echo "Example: $0 config_infer_yolov8.txt"
    exit 1
fi

CONFIG_NAME="$1"

if [[ ! -f "config/$CONFIG_NAME" ]]; then
    echo "Error: config/$CONFIG_NAME not found"
    exit 1
fi

echo "Pre-building TensorRT engine for $CONFIG_NAME..."
echo ""

# Run trtexec or use deepstream-app in headless mode to build engine
docker run --rm \
  --gpus all \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/libnvdsinfer_custom_impl_Yolo.so":/app/libnvdsinfer_custom_impl_Yolo.so \
  nvcr.io/nvidia/deepstream:8.0-triton-multiarch \
  bash -c "
    echo 'Building engine from config: /config/$CONFIG_NAME'
    # Create minimal test config
    cat > /tmp/test.txt <<EOF
[application]
enable-perf-measurement=0

[source0]
enable=1
type=3
uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
num-sources=1

[sink0]
enable=1
type=1

[streammux]
width=1920
height=1080
batch-size=1

[primary-gie]
enable=1
config-file=/config/$CONFIG_NAME

[osd]
enable=0
EOF
    # Run for 10 seconds to build engine
    timeout 15 deepstream-app -c /tmp/test.txt || true
    echo ''
    echo 'Engine built! Checking models directory:'
    ls -lh /models/*.engine 2>/dev/null || echo 'No engines found'
  "

echo ""
echo "✓ Engine cached in models/ directory"
ls -lh models/*.engine 2>/dev/null || echo "Warning: No .engine files found"
