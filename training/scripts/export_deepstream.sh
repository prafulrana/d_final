#!/bin/bash
# Export YOLOv8 model to DeepStream-compatible ONNX format using Ultralytics container
# Usage: ./scripts/export_deepstream.sh <model.pt path> <size>
# Example: ./scripts/export_deepstream.sh bowling/bowling_640_blur_v3/weights/best.pt 640

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model.pt path> <size>"
    echo "Example: $0 bowling/bowling_640_blur_v3/weights/best.pt 640"
    exit 1
fi

MODEL_PATH="$1"
SIZE="$2"

# Verify model exists
if [ ! -f "/root/d_final/training/$MODEL_PATH" ]; then
    echo "Error: Model not found at /root/d_final/training/$MODEL_PATH"
    exit 1
fi

echo "=================================================="
echo "DeepStream-Yolo Export Script"
echo "=================================================="
echo "Model: $MODEL_PATH"
echo "Size: $SIZE"
echo "Base image: ultralytics/ultralytics:latest"
echo "=================================================="

# Run export in Ultralytics container with DeepStream-Yolo export script
docker run --rm --gpus all \
  -v /root/d_final/training:/data \
  ultralytics/ultralytics:latest \
  bash -c "pip install -q onnx onnxslim && cd /data && python3 DeepStream-Yolo/utils/export_yoloV8.py -w $MODEL_PATH --dynamic -s $SIZE"

echo ""
echo "Export completed. Checking for ONNX output..."
find /root/d_final/training -name "*.onnx" -mmin -1 2>/dev/null | head -5
