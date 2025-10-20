#!/bin/bash
# Publisher: Continuously streams test video to relay's in_s2

cd "$(dirname "$0")"

# Stop and remove existing publisher container
docker stop publisher 2>/dev/null
docker rm publisher 2>/dev/null

# Start publisher container
docker run -d \
  --name publisher \
  --net=host \
  publisher:latest

echo "Publisher started. Streaming to in_s2 at relay."
echo "Check logs with: docker logs publisher -f"
