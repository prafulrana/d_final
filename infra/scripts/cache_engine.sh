#!/bin/bash
# TensorRT Engine Cache Management
# Copies engines from container /app/ (ephemeral) to /models/ (bind-mounted)

case "$1" in
    copy)
        echo "Copying TensorRT engines to /models/ for persistence..."

        # s0: COCO YOLOv8n
        if docker exec ds-s0 test -f /app/model_b1_gpu0_fp16.engine 2>/dev/null; then
            docker cp ds-s0:/app/model_b1_gpu0_fp16.engine /root/d_final/models/yolov8n_b1_gpu0_fp16.engine
            echo "✓ s0: yolov8n_b1_gpu0_fp16.engine ($(du -h /root/d_final/models/yolov8n_b1_gpu0_fp16.engine | cut -f1))"
        else
            echo "✗ s0: Engine not found"
        fi

        # s1: Bowling ball 640 blur
        if docker exec ds-s1 test -f /app/model_b1_gpu0_fp16.engine 2>/dev/null; then
            docker cp ds-s1:/app/model_b1_gpu0_fp16.engine /root/d_final/models/bowling_640_blur_b1_gpu0_fp16.engine
            echo "✓ s1: bowling_640_blur_b1_gpu0_fp16.engine ($(du -h /root/d_final/models/bowling_640_blur_b1_gpu0_fp16.engine | cut -f1))"
        else
            echo "✗ s1: Engine not found"
        fi

        # s2: Bowling ball 640 blur (same as s1, verify cached)
        if [ -f /root/d_final/models/bowling_640_blur_b1_gpu0_fp16.engine ]; then
            echo "✓ s2: bowling_640_blur_b1_gpu0_fp16.engine (shared with s1)"
        else
            echo "✗ s2: Engine missing (should be cached from s1)"
        fi

        # s3: Bowling pins 640
        if docker exec ds-s3 test -f /app/model_b1_gpu0_fp16.engine 2>/dev/null; then
            docker cp ds-s3:/app/model_b1_gpu0_fp16.engine /root/d_final/models/bowling_pins_640_b1_gpu0_fp16.engine
            echo "✓ s3: bowling_pins_640_b1_gpu0_fp16.engine ($(du -h /root/d_final/models/bowling_pins_640_b1_gpu0_fp16.engine | cut -f1))"
        else
            echo "✗ s3: Engine not found"
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
            if docker ps --filter "name=$c" --format "{{.Names}}" | grep -q "$c" 2>/dev/null; then
                echo "$c:"
                docker exec $c find /app -name "*.engine" -type f -ls 2>/dev/null | awk '{print "  " $11 " (" $7/1024/1024 " MB)"}' || echo "  (none)"
            fi
        done
        ;;

    verify)
        # Verify configs point to existing cached engines
        echo "=== Verifying config engine paths ==="

        configs=(
            "s0_coco_1280.txt:yolov8n_b1_gpu0_fp16.engine"
            "s1_bowling_ball_640.txt:bowling_640_blur_b1_gpu0_fp16.engine"
            "s2_bowling_ball_640.txt:bowling_640_blur_b1_gpu0_fp16.engine"
            "s3_bowling_pins_640.txt:bowling_pins_640_b1_gpu0_fp16.engine"
        )

        for entry in "${configs[@]}"; do
            config="${entry%%:*}"
            engine="${entry##*:}"

            printf "%-30s " "[$config]"
            if grep -q "model-engine-file=/models/$engine" /root/d_final/config/$config 2>/dev/null; then
                if [ -f /root/d_final/models/$engine ]; then
                    echo "✓ /models/$engine (exists)"
                else
                    echo "✗ /models/$engine (MISSING - run './cache_engine.sh copy')"
                fi
            else
                echo "⚠ Doesn't point to expected engine"
            fi
        done
        ;;

    clean)
        echo "Deleting cached engines from /models/..."
        rm -f /root/d_final/models/*.engine
        echo "✓ Engines deleted"
        ;;

    *)
        echo "Usage: $0 {copy|list|verify|clean}"
        echo ""
        echo "  copy   - Copy engines from containers to /models/ for persistence"
        echo "  list   - Show engines in /models/ and containers"
        echo "  verify - Verify configs point to cached engines"
        echo "  clean  - Delete cached engines from /models/"
        exit 1
        ;;
esac
