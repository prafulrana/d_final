#!/bin/bash
# TensorRT Engine Cache Management
# Copies engines from container /app/ (ephemeral) to /models/ (bind-mounted)

set -e

case "$1" in
    copy)
        echo "Copying TensorRT engines to /models/ for persistence..."

        # Copy COCO YOLOv8n engine (from s0 or s1)
        if docker exec ds-s0 test -f /app/model_b1_gpu0_fp16.engine 2>/dev/null; then
            docker cp ds-s0:/app/model_b1_gpu0_fp16.engine /root/d_final/models/yolov8n_b1_gpu0_fp16.engine
            echo "✓ Copied COCO engine: yolov8n_b1_gpu0_fp16.engine ($(du -h /root/d_final/models/yolov8n_b1_gpu0_fp16.engine | cut -f1))"
        else
            echo "✗ COCO engine not found in ds-s0"
        fi

        # Copy Bowling Roboflow engine (from s2 or s3)
        if docker exec ds-s2 test -f /app/model_b1_gpu0_fp16.engine 2>/dev/null; then
            docker cp ds-s2:/app/model_b1_gpu0_fp16.engine /root/d_final/models/bowling_roboflow_1280_b1_gpu0_fp16.engine
            echo "✓ Copied Bowling engine: bowling_roboflow_1280_b1_gpu0_fp16.engine ($(du -h /root/d_final/models/bowling_roboflow_1280_b1_gpu0_fp16.engine | cut -f1))"
        else
            echo "✗ Bowling engine not found in ds-s2"
        fi

        echo ""
        echo "Cached engines in /models/:"
        ls -lh /root/d_final/models/*.engine 2>/dev/null || echo "  (none)"
        ;;

    list)
        echo "=== Engines in /models/ (bind-mounted, persistent) ==="
        ls -lh /root/d_final/models/*.engine 2>/dev/null || echo "  (none)"
        echo ""
        echo "=== Engines in containers (ephemeral) ==="
        for c in ds-s0 ds-s1 ds-s2 ds-s3; do
            if docker ps --filter "name=$c" --format "{{.Names}}" | grep -q "$c"; then
                echo "$c:"
                docker exec $c find /app -name "*.engine" -type f -ls 2>/dev/null | awk '{print "  " $11 " (" $7/1024/1024 " MB)"}' || echo "  (none)"
            fi
        done
        ;;

    clean)
        echo "Deleting cached engines from /models/..."
        rm -f /root/d_final/models/*.engine
        echo "✓ Engines deleted"
        ;;

    *)
        echo "Usage: $0 {copy|list|clean}"
        echo ""
        echo "  copy  - Copy engines from containers to /models/ for persistence"
        echo "  list  - Show engines in /models/ and containers"
        echo "  clean - Delete cached engines from /models/"
        exit 1
        ;;
esac
