#!/bin/bash
# Download pre-trained models from various sources
# Usage: ./download_model.sh <model_type> <model_name>

set -e

MODEL_TYPE=${1}
MODEL_NAME=${2}
MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"

case "$MODEL_TYPE" in
    yolov8)
        echo "Downloading YOLOv8 model: ${MODEL_NAME}"
        docker run --rm -v "$MODELS_DIR":/workspace \
            ultralytics/ultralytics:latest bash -c "
                cd /workspace && \
                yolo export model=${MODEL_NAME}.pt format=pytorch
            "
        echo "✓ Downloaded: ${MODEL_NAME}.pt"
        ;;

    ngc)
        # NGC models (PeopleSemSegNet, PeopleSegNet, etc.)
        echo "Downloading from NGC: ${MODEL_NAME}"
        NGC_CLI="/root/d_final/ngc-cli/ngc"

        if [ ! -f "$NGC_CLI" ]; then
            echo "✗ NGC CLI not found. Run: ./download_ngc_cli.sh first"
            exit 1
        fi

        cd "$MODELS_DIR"
        $NGC_CLI registry model download-version "$MODEL_NAME"
        echo "✓ Downloaded NGC model: ${MODEL_NAME}"
        ;;

    roboflow)
        echo "Roboflow models require manual download"
        echo "Visit your Roboflow project and download the model"
        echo "Then place the ONNX file in: $MODELS_DIR"
        ;;

    *)
        echo "Usage: $0 <yolov8|ngc|roboflow> <model_name>"
        echo ""
        echo "Examples:"
        echo "  $0 yolov8 yolov8n"
        echo "  $0 yolov8 yolov8s"
        echo "  $0 ngc nvidia/tao/peoplesegnet:deployable_v2.0.2"
        exit 1
        ;;
esac
