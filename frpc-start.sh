#!/bin/bash
# Start frpc tunnel client

CONFIG="/root/d_final/config/frpc.ini"
LOG="/var/log/frpc.log"

# Check if already running
if pgrep -x frpc > /dev/null; then
    echo "frpc is already running (PID: $(pgrep -x frpc))"
    exit 1
fi

# Start frpc
echo "Starting frpc..."
nohup frpc -c "$CONFIG" > "$LOG" 2>&1 &
sleep 2

# Verify started
if pgrep -x frpc > /dev/null; then
    PID=$(pgrep -x frpc)
    echo "✓ frpc started (PID: $PID)"

    # Wait for connection
    echo "Waiting for connection to relay..."
    for i in {1..10}; do
        if grep -q "login to server success" "$LOG" 2>/dev/null; then
            echo "✓ Connected to relay"
            echo "✓ Tunnels active: $(grep -oP 'proxy added: \K\[.*\]' "$LOG" | tail -1)"
            exit 0
        fi
        sleep 1
    done

    echo "⚠ frpc started but connection not confirmed"
    echo "Check logs: tail -20 $LOG"
    exit 0
else
    echo "✗ Failed to start frpc"
    echo "Check config: $CONFIG"
    exit 1
fi
