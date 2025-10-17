#!/bin/bash
# Manage TensorRT engine cache between containers and host
# Usage: ./cache_engine.sh <command> [container_name]

set -e

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
COMMAND=${1:-list}
CONTAINER=${2:-drishti-s0}

case "$COMMAND" in
    list)
        echo "=== Checking TensorRT engines in containers ==="
        for container in drishti-s0 drishti-s1 drishti-s2; do
            if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
                echo ""
                echo "Container: $container"
                docker exec "$container" find /models -name "*.engine" -exec ls -lh {} \; 2>/dev/null || echo "  (container not running or no engines found)"
            fi
        done

        echo ""
        echo "=== TensorRT engines on host ==="
        ls -lh "$MODELS_DIR"/*.engine 2>/dev/null || echo "No .engine files in $MODELS_DIR"
        ;;

    copy)
        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
            echo "✗ Container $CONTAINER is not running"
            echo "Available containers:"
            docker ps --format '{{.Names}}' | grep drishti || echo "(none)"
            exit 1
        fi

        echo "Copying TensorRT engines from $CONTAINER to host..."

        # Find all .engine files in container's /models directory
        ENGINE_FILES=$(docker exec "$CONTAINER" find /models -name "*.engine" 2>/dev/null || true)

        if [ -z "$ENGINE_FILES" ]; then
            echo "✗ No .engine files found in $CONTAINER:/models/"
            echo ""
            echo "Possible reasons:"
            echo "  1. Engine still building (check logs: docker logs -f $CONTAINER)"
            echo "  2. Pipeline hasn't started yet"
            echo "  3. Engine building failed"
            exit 1
        fi

        for engine in $ENGINE_FILES; do
            engine_name=$(basename "$engine")
            echo "  Copying: $engine_name"
            docker cp "$CONTAINER:$engine" "$MODELS_DIR/$engine_name"
            echo "  ✓ Saved to: $MODELS_DIR/$engine_name"
        done

        echo ""
        echo "✓ Engine cache copied successfully!"
        echo ""
        echo "Cached engines:"
        ls -lh "$MODELS_DIR"/*.engine
        ;;

    verify)
        echo "=== Verifying TensorRT engine cache ==="

        # Read config to find expected engine path
        CONFIG_FILE="${3:-$MODELS_DIR/../config/pgie_yolov8_coco.txt}"

        if [ ! -f "$CONFIG_FILE" ]; then
            echo "✗ Config file not found: $CONFIG_FILE"
            exit 1
        fi

        ENGINE_PATH=$(grep "model-engine-file" "$CONFIG_FILE" | cut -d= -f2)
        ENGINE_FILE=$(basename "$ENGINE_PATH")
        HOST_ENGINE_PATH="$MODELS_DIR/$ENGINE_FILE"

        echo "Config file: $CONFIG_FILE"
        echo "Expected engine: $ENGINE_FILE"
        echo ""

        if [ -f "$HOST_ENGINE_PATH" ]; then
            ENGINE_SIZE=$(ls -lh "$HOST_ENGINE_PATH" | awk '{print $5}')
            ENGINE_AGE=$(stat -c %y "$HOST_ENGINE_PATH" | cut -d. -f1)
            echo "✓ Engine exists: $HOST_ENGINE_PATH"
            echo "  Size: $ENGINE_SIZE"
            echo "  Modified: $ENGINE_AGE"
            echo ""
            echo "✓ Pipeline will use cached engine (instant startup)"
        else
            echo "✗ Engine not cached: $HOST_ENGINE_PATH"
            echo ""
            echo "First pipeline start will build engine (~5-10 min for 2048x2048)"
            echo "After building, run: ./scripts/cache_engine.sh copy"
        fi
        ;;

    clean)
        echo "=== Cleaning old TensorRT engines ==="
        echo "This will delete all .engine files from host models/ directory"
        read -p "Continue? (y/N) " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$MODELS_DIR"/*.engine
            echo "✓ Cleaned all engine files"
        else
            echo "Cancelled"
        fi
        ;;

    *)
        echo "TensorRT Engine Cache Management"
        echo ""
        echo "Usage: $0 <command> [container_name]"
        echo ""
        echo "Commands:"
        echo "  list              List all engines in containers and on host"
        echo "  copy [container]  Copy engines from container to host (default: drishti-s0)"
        echo "  verify [config]   Check if engine exists for given config"
        echo "  clean             Delete all cached engines from host"
        echo ""
        echo "Examples:"
        echo "  $0 list"
        echo "  $0 copy drishti-s0"
        echo "  $0 verify config/pgie_yolov8_coco.txt"
        echo "  $0 clean"
        echo ""
        echo "Workflow:"
        echo "  1. Start pipelines: ./up.sh"
        echo "  2. Wait for engine build (5-10 min): docker logs -f drishti-s0"
        echo "  3. Copy engine to host: $0 copy drishti-s0"
        echo "  4. Future runs use cached engine (instant startup)"
        exit 1
        ;;
esac
