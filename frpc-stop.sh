#!/bin/bash
# Stop frpc tunnel client

if ! pgrep -x frpc > /dev/null; then
    echo "frpc is not running"
    exit 0
fi

PID=$(pgrep -x frpc)
echo "Stopping frpc (PID: $PID)..."
pkill -x frpc

# Wait for process to die
for i in {1..5}; do
    if ! pgrep -x frpc > /dev/null; then
        echo "✓ frpc stopped"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if pgrep -x frpc > /dev/null; then
    echo "Force killing frpc..."
    pkill -9 -x frpc
    sleep 1

    if ! pgrep -x frpc > /dev/null; then
        echo "✓ frpc force stopped"
        exit 0
    else
        echo "✗ Failed to stop frpc"
        exit 1
    fi
fi
