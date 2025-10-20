#!/bin/bash
# Check frpc tunnel client status

LOG="/var/log/frpc.log"

echo "=== frpc Status ==="
echo ""

# Check if running
if pgrep -x frpc > /dev/null; then
    PID=$(pgrep -x frpc)
    echo "✓ Process: Running (PID: $PID)"

    # Check connection (look in full log for last successful login)
    LAST_LOGIN=$(grep "login to server success" "$LOG" 2>/dev/null | tail -1)
    if [ -n "$LAST_LOGIN" ]; then
        echo "✓ Connection: Connected to relay"

        # Show tunnels
        TUNNELS=$(grep -oP 'proxy added: \K\[.*\]' "$LOG" | tail -1)
        if [ -n "$TUNNELS" ]; then
            echo "✓ Tunnels: $TUNNELS"
        fi

        # Show recent errors (last 20 lines)
        ERROR_COUNT=$(tail -20 "$LOG" 2>/dev/null | grep -c "\[E\]")
        if [ "$ERROR_COUNT" -gt 0 ]; then
            echo "⚠ Recent errors: $ERROR_COUNT in last 20 lines"
            echo ""
            echo "Recent log tail:"
            tail -10 "$LOG" | sed 's/^/  /'
        else
            echo "✓ No recent errors"
        fi
    else
        echo "✗ Connection: Not connected (check logs)"
        echo ""
        echo "Recent log tail:"
        tail -10 "$LOG" 2>/dev/null | sed 's/^/  /' || echo "  (no log file)"
    fi
else
    echo "✗ Process: Not running"
    echo ""
    echo "Start with: ./frpc-start.sh"
fi

echo ""
echo "Full logs: tail -f $LOG"
