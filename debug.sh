#!/bin/bash
# Consolidated debugging for DeepStream system

echo "=== DeepStream Debug Report ==="
echo "Generated: $(date)"
echo ""

# Containers
echo "[Containers]"
docker ps -a --filter "name=ds-s" --filter "name=publisher" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# GPU
echo "[GPU]"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
  awk -F',' '{printf "  GPU %s: %s%% util, %s/%s MiB\n", $1, $3, $4, $5}'
echo ""

# RTSP Servers
echo "[RTSP Servers]"
for i in 0 1 2; do
    PORT=$((8554 + i))
    if ss -tlnp | grep -q ":${PORT} "; then
        echo "  ✓ s${i} on ${PORT}"
    else
        echo "  ✗ s${i} on ${PORT} NOT listening"
    fi
done
echo ""

# frpc
echo "[frpc Tunnel]"
if pgrep -x frpc > /dev/null; then
    echo "  PID: $(pgrep -x frpc)"
    if grep -q "login to server success" /var/log/frpc.log 2>/dev/null; then
        echo "  ✓ Connected"
        grep -oP 'proxy added: \K\[.*\]' /var/log/frpc.log | tail -1 | sed 's/^/  Tunnels: /'
    else
        echo "  ✗ Not connected"
    fi
    echo "  Recent errors:"
    tail -10 /var/log/frpc.log | grep "\[E\]" | sed 's/^/    /' || echo "    (none)"
else
    echo "  ✗ Not running"
fi
echo ""

# Container logs
for container in ds-s0 ds-s1 ds-s2; do
    echo "[$container Logs (last 5 lines)]"
    docker logs "$container" --tail 5 2>&1 | sed 's/^/  /' || echo "  (container not found)"
    echo ""
done

echo "=== End Debug Report ==="
echo ""
echo "For live logs: docker logs -f ds-s0"
echo "For frpc logs: tail -f /var/log/frpc.log"
