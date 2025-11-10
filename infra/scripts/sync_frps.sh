#!/bin/bash
# Sync frps config to relay VM and restart frps

RELAY="mediamtx-relay"
ZONE="asia-south1-c"
LOCAL_CONFIG="/root/d_final/infra/configs/frps.ini"
REMOTE_CONFIG="/etc/frp/frps.ini"

echo "Syncing frps config to relay..."

# Copy config to relay
/home/prafulrana/google-cloud-sdk/bin/gcloud compute scp "$LOCAL_CONFIG" "$RELAY:$REMOTE_CONFIG" --zone="$ZONE"

if [ $? -eq 0 ]; then
    echo "✓ Config copied"

    # Restart frps
    echo "Restarting frps container..."
    /home/prafulrana/google-cloud-sdk/bin/gcloud compute ssh "$RELAY" --zone="$ZONE" \
      --command="docker restart frps && sleep 3 && docker logs frps --tail 10"

    echo ""
    echo "✓ frps restarted"
else
    echo "✗ Failed to copy config"
    exit 1
fi
