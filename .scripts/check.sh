#!/bin/bash
# Health check script for DeepStream pipeline
# Validates all components are working correctly

echo "=== DeepStream Health Check ==="
echo ""

FAIL=0

# Check 1: Containers running
echo "[1/6] Checking containers..."
EXPECTED_CONTAINERS=("ds-s0" "ds-s1" "ds-s2")
for container in "${EXPECTED_CONTAINERS[@]}"; do
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "  ✓ $container running"
    else
        echo "  ✗ $container NOT running"
        FAIL=1
    fi
done
echo ""

# Check 2: TensorRT engine exists
echo "[2/6] Checking TensorRT engine..."
if [ -f "models/yolov8n_b1_gpu0_fp16.engine" ]; then
    SIZE=$(du -h models/yolov8n_b1_gpu0_fp16.engine | cut -f1)
    echo "  ✓ Engine cached ($SIZE)"
else
    echo "  ✗ Engine NOT found (will build on first run)"
    FAIL=1
fi
echo ""

# Check 3: RTSP servers responding
echo "[3/6] Checking RTSP servers..."
for i in 0 1 2; do
    PORT=$((8554 + i))
    if ss -tlnp | grep -q ":${PORT} "; then
        echo "  ✓ s${i} RTSP server listening on ${PORT}"
    else
        echo "  ✗ s${i} RTSP server NOT on ${PORT}"
        FAIL=1
    fi
done
echo ""

# Check 4: frpc tunnel
echo "[4/6] Checking frpc tunnel..."
if pgrep -x frpc > /dev/null; then
    echo "  ✓ frpc process running"
    if tail -20 /var/log/frpc.log 2>/dev/null | grep -q "login to server success"; then
        echo "  ✓ frpc connected to relay"
    else
        echo "  ⚠ frpc running but may not be connected (check logs)"
        FAIL=1
    fi
else
    echo "  ✗ frpc NOT running"
    FAIL=1
fi
echo ""

# Check 5: DeepStream pipelines initialized
echo "[5/6] Checking DeepStream pipelines..."
for i in 0 1 2; do
    if docker logs ds-s${i} 2>&1 | grep -q "Pipeline set to PLAYING"; then
        echo "  ✓ s${i} pipeline PLAYING"
    else
        echo "  ✗ s${i} pipeline NOT ready"
        FAIL=1
    fi
done
echo ""

# Check 6: GPU usage
echo "[6/6] Checking GPU..."
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -ge 3 ]; then
    echo "  ✓ GPU in use ($GPU_PROCS processes)"
else
    echo "  ⚠ GPU usage looks low ($GPU_PROCS processes)"
fi
echo ""

# Summary
echo "========================="
if [ $FAIL -eq 0 ]; then
    echo "✓ All checks PASSED"
    echo "System is healthy"
    exit 0
else
    echo "✗ Some checks FAILED"
    echo "Run ./debug.sh for detailed troubleshooting"
    exit 1
fi
