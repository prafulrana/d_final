#!/bin/bash
# Sync mediamtx config to relay VM and restart mediamtx

RELAY="mediamtx-relay"
ZONE="asia-south1-c"
LOCAL_CONFIG="/root/d_final/infra/configs/mediamtx.yml"
REMOTE_CONFIG="/etc/mediamtx/config.yml"

echo "Syncing mediamtx config to relay..."

# Copy config to relay
/home/prafulrana/google-cloud-sdk/bin/gcloud compute scp "$LOCAL_CONFIG" "$RELAY:$REMOTE_CONFIG" --zone="$ZONE"

if [ $? -eq 0 ]; then
    echo "✓ Config copied"

    # Restart mediamtx
    echo "Restarting mediamtx container..."
    /home/prafulrana/google-cloud-sdk/bin/gcloud compute ssh "$RELAY" --zone="$ZONE" \
      --command="docker restart mediamtx && sleep 3 && docker logs mediamtx --tail 10"

    echo ""
    echo "✓ mediamtx restarted"
else
    echo "✗ Failed to copy config"
    exit 1
fi
