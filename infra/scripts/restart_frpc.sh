#!/bin/bash
# Restart local frpc with updated config

FRPC_CONFIG="/root/d_final/infra/scripts/frpc/frpc.ini"
FRPC_LOG="/var/log/frpc.log"

echo "Restarting frpc..."

# Stop existing frpc
if pgrep -x frpc > /dev/null; then
    echo "Stopping existing frpc process..."
    pkill -x frpc
    sleep 2
fi

# Start frpc
echo "Starting frpc with config: $FRPC_CONFIG"
nohup frpc -c "$FRPC_CONFIG" > "$FRPC_LOG" 2>&1 &

sleep 3

# Check if started
if pgrep -x frpc > /dev/null; then
    if grep -q "login to server success" "$FRPC_LOG" 2>/dev/null; then
        echo "✓ frpc started and connected"
        echo ""
        echo "Recent logs:"
        tail -5 "$FRPC_LOG"
    else
        echo "⚠ frpc started but not yet connected (check logs)"
        tail -10 "$FRPC_LOG"
    fi
else
    echo "✗ frpc failed to start"
    tail -10 "$FRPC_LOG"
    exit 1
fi
