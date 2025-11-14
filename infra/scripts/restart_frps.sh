#!/bin/bash
# Restart frps on relay to pick up new frpc tunnels

RELAY="mediamtx-relay"
ZONE="asia-south1-c"

echo "Restarting frps on relay..."
/home/prafulrana/google-cloud-sdk/bin/gcloud compute ssh "$RELAY" --zone="$ZONE" \
  --command="docker restart frps && sleep 3 && docker logs frps --tail 10"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ frps restarted"
else
    echo "✗ Failed to restart frps"
    exit 1
fi
