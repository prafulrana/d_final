#!/bin/bash
# Update inference model config across s0_rtsp.py and file_s3_s4.txt
# Usage: ./update_model.sh <config_name>
# Example: ./update_model.sh config_infer_yoloworld.txt

set -e

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <config_name>"
    echo ""
    echo "Available configs:"
    echo "  config_infer_primary.txt    - TrafficCamNet (4 classes)"
    echo "  config_infer_yoloworld.txt  - YOLOWorld (80 COCO classes)"
    echo ""
    echo "Example: $0 config_infer_yoloworld.txt"
    exit 1
fi

CONFIG_NAME="$1"
PYTHON_SCRIPT="s0_rtsp.py"
S3S4_CONFIG="config/file_s3_s4.txt"

if [[ ! -f "config/$CONFIG_NAME" ]]; then
    echo "Error: config/$CONFIG_NAME not found"
    exit 1
fi

echo "Switching to $CONFIG_NAME..."

# Update Python script (fix the set_property line)
sed -i "s|pgie.set_property('config-file-path', \"/config/config_infer[^\"]*\.txt\")|pgie.set_property('config-file-path', \"/config/$CONFIG_NAME\")|" "$PYTHON_SCRIPT"

# Update s3/s4 config
sed -i "s|config-file=/config/config_infer.*\.txt|config-file=/config/$CONFIG_NAME|" "$S3S4_CONFIG"

echo "✓ Inference config synchronized!"
echo ""
echo "Active model: $CONFIG_NAME"
grep "config-file-path\|config-file=" "$PYTHON_SCRIPT" "$S3S4_CONFIG" | grep -v "^#"
echo ""
echo "Next steps:"
echo "  1. Pre-build engine (optional, saves startup time):"
echo "     ./cache_engine.sh $CONFIG_NAME"
echo ""
echo "  2. Or delete old engines and restart:"
echo "     rm models/*.engine"
echo "     ./start.sh"
