#!/usr/bin/env bash
set -euo pipefail

PATH_REGEX='${path_regex}'

echo "[startup] Installing prerequisites..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y docker.io curl

mkdir -p /etc/mediamtx

echo "[startup] Detecting public IP from metadata..."
PUBLIC_IP="$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip || echo 127.0.0.1)"

cat >/etc/mediamtx/config.yml <<EOF
# Logging
logLevel: info

# RTSP configuration
rtsp: yes

# HLS / RTMP (defaults enabled)
hls: yes
rtmp: yes

# WebRTC configuration - reduced gather timeouts for faster connection
webrtc: yes
webrtcAddress: ":8889"
webrtcEncryption: no
webrtcAllowOrigin: "*"
webrtcLocalUDPAddress: ":8189"
webrtcLocalTCPAddress: ":8189"
webrtcIPsFromInterfaces: no
webrtcAdditionalHosts: ["${PUBLIC_IP}"]
webrtcICEServers2: []
webrtcHandshakeTimeout: 5s
webrtcTrackGatherTimeout: 1s
webrtcSTUNGatherTimeout: 2s

# API configuration
api: yes
apiAddress: ":9997"

# Path configuration - allow all paths
paths:
  all: {}
EOF

echo "[startup] Starting MediaMTX container..."
docker rm -f mediamtx >/dev/null 2>&1 || true
docker run -d \
  --name mediamtx \
  --restart unless-stopped \
  --network host \
  -v /etc/mediamtx/config.yml:/mediamtx.yml:ro \
  bluenviron/mediamtx:latest

echo "[startup] Done. Logs: docker logs -f mediamtx"

