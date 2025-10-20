#!/bin/bash
# Restart frpc tunnel client

echo "=== Restarting frpc ==="
echo ""

# Stop
./frpc-stop.sh
echo ""

# Start
./frpc-start.sh
