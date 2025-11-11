#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <stream_id>"
    echo "Example: $0 0  # Restarts source 0 (in_s0)"
    echo "Example: $0 1  # Restarts source 1 (in_s1)"
    echo "Example: $0 2  # Restarts source 2 (in_s2)"
    exit 1
fi

STREAM_ID=$1
API_KEY="bowl2024"

echo "Restarting stream ${STREAM_ID} via relay..."
curl -X POST http://34.47.221.242:5555/stream/restart \
    -H "X-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"id\": ${STREAM_ID}}"

echo ""
