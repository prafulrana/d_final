#!/bin/bash
# Manage remote relay (frps + mediamtx)

RELAY="mediamtx-relay"
ZONE="asia-south1-c"

case "$1" in
    restart)
        /home/prafulrana/google-cloud-sdk/bin/gcloud compute ssh "$RELAY" --zone="$ZONE" \
          --command="docker restart frps mediamtx && sleep 3 && docker ps --filter 'name=frps|mediamtx'"
        ;;
    status)
        /home/prafulrana/google-cloud-sdk/bin/gcloud compute ssh "$RELAY" --zone="$ZONE" \
          --command="docker ps --filter 'name=frps|mediamtx' && echo '' && docker logs mediamtx --tail 10"
        ;;
    *)
        echo "Usage: $0 {restart|status}"
        exit 1
        ;;
esac
