#!/bin/bash
# Export YOLOv8 model to DeepStream-compatible ONNX format
# Usage: ./export_yolov8.sh <model_name> <resolution>
# Example: ./export_yolov8.sh yolov8n 2048

set -e

MODEL=${1:-yolov8n}
RESOLUTION=${2:-1024}
MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
UTILS_DIR="$(cd "$(dirname "$0")/.." && pwd)/utils"

echo "Exporting ${MODEL} at ${RESOLUTION}x${RESOLUTION}"
echo "Models dir: $MODELS_DIR"
echo "Utils dir: $UTILS_DIR"

# Download model if not exists
if [ ! -f "$MODELS_DIR/${MODEL}.pt" ]; then
    echo "Downloading ${MODEL}.pt..."
    docker run --rm -v "$MODELS_DIR":/models \
        ultralytics/ultralytics:latest \
        yolo export model=${MODEL}.pt format=pytorch
fi

# Export to ONNX with DeepStream format
echo "Converting to ONNX..."
docker run --rm \
    -v "$UTILS_DIR":/utils \
    -v "$MODELS_DIR":/models \
    ultralytics/ultralytics:latest bash -c "
        pip install -q onnx onnxsim && \
        cd /models && \
        python3 /utils/export_yoloV8.py -w ${MODEL}.pt --dynamic -s ${RESOLUTION}
    "

# Rename output
if [ -f "$MODELS_DIR/${MODEL}.pt.onnx" ]; then
    mv "$MODELS_DIR/${MODEL}.pt.onnx" "$MODELS_DIR/${MODEL}_${RESOLUTION}.onnx"
    echo "✓ Exported: ${MODEL}_${RESOLUTION}.onnx"
else
    echo "✗ Export failed"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Update config file to use: onnx-file=/models/${MODEL}_${RESOLUTION}.onnx"
echo "2. Update engine file path: model-engine-file=/models/${MODEL}_${RESOLUTION}_b1_gpu0_fp16.engine"
echo "3. Run ./build.sh && ./up.sh to deploy"
