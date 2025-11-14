#!/bin/bash
# Restart both frps and mediamtx on relay

RELAY="mediamtx-relay"
ZONE="asia-south1-c"

echo "Restarting frps and mediamtx on relay..."
/home/prafulrana/google-cloud-sdk/bin/gcloud compute ssh "$RELAY" --zone="$ZONE" \
  --command="docker restart frps mediamtx && sleep 3 && docker ps --filter 'name=frps|mediamtx'"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ frps and mediamtx restarted"
else
    echo "✗ Failed to restart relay services"
    exit 1
fi
