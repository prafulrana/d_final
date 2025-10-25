#!/bin/bash
# Export pretrained YOLO COCO model to DeepStream-compatible ONNX format
# Usage: ./scripts/export_coco.sh <model> <size>
# Example: ./scripts/export_coco.sh yolo12x 1280

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model> <size>"
    echo "Example: $0 yolo12x 1280"
    echo "Example: $0 yolov8n 640"
    exit 1
fi

MODEL="$1"
SIZE="$2"

echo "=================================================="
echo "DeepStream-Yolo COCO Export Script"
echo "=================================================="
echo "Model: ${MODEL}.pt (pretrained COCO)"
echo "Size: ${SIZE}x${SIZE}"
echo "Base image: ultralytics/ultralytics:latest"
echo "=================================================="

# Run export in custom training container (has onnx/onnxslim pre-installed)
# model.pt will be auto-downloaded if not present
docker run --rm --gpus all \
  -v /root/d_final/training:/data \
  ds-training:latest \
  bash -c "cd /data && python3 DeepStream-Yolo/utils/export_yoloV8.py -w ${MODEL}.pt --dynamic -s $SIZE"

echo ""
echo "Export completed. Output:"
find /root/d_final/training -name "${MODEL}.onnx" -o -name "${MODEL}.pt.onnx" -mmin -1 2>/dev/null
