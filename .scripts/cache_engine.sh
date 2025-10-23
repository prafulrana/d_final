#!/bin/bash
# Auto-cache TensorRT engines from /app/ to /models/ for persistence

set -e

ACTION=${1:-"copy"}
CONTAINER=${2:-"ds-s3"}

case "$ACTION" in
    copy)
        echo "Copying engines from container to mounted volume..."
        docker exec "$CONTAINER" bash -c '
            for engine in /app/*.engine; do
                if [ -f "$engine" ]; then
                    filename=$(basename "$engine")
                    echo "Found: $filename"
                    cp "$engine" /models/ 2>/dev/null || true
                fi
            done
        '
        echo "✓ Engine cache updated"
        ls -lh /root/d_final/models/*.engine
        ;;

    list)
        echo "=== Engines in container ($CONTAINER) ==="
        docker exec "$CONTAINER" ls -lh /app/*.engine 2>/dev/null || echo "No engines in /app/"
        echo ""
        echo "=== Cached engines on host ==="
        ls -lh /root/d_final/models/*.engine 2>/dev/null || echo "No cached engines"
        ;;

    clean)
        echo "Removing all cached engines..."
        rm -f /root/d_final/models/*.engine
        echo "✓ Cache cleared"
        ;;

    *)
        echo "Usage: $0 {copy|list|clean} [container_name]"
        echo ""
        echo "Commands:"
        echo "  copy  - Copy engines from container to host (default)"
        echo "  list  - List engines in container and on host"
        echo "  clean - Remove all cached engines"
        exit 1
        ;;
esac
