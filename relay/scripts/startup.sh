#!/usr/bin/env bash
set -euo pipefail

PATH_REGEX='${path_regex}'
FRP_TOKEN='${frp_token}'

echo "[startup] Installing prerequisites..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y docker.io curl

mkdir -p /etc/mediamtx
mkdir -p /etc/frp

echo "[startup] Detecting public IP from metadata..."
PUBLIC_IP="$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip || echo 127.0.0.1)"

cat >/etc/frp/frps.ini <<EOF
[common]
bind_port = 7000
token = $${FRP_TOKEN}
tcp_mux = true
allow_ports = 9500-9600
max_pool_count = 32
log_file = /var/log/frps.log
log_level = info
log_max_days = 3
EOF

cat >/etc/mediamtx/config.yml <<EOF
# Logging
logLevel: info

# RTSP / HLS / RTMP (defaults enabled by the container)
rtsp: yes
hls: yes
rtmp: yes

# WebRTC configuration
webrtc: yes
webrtcAddress: ":8889"
webrtcEncryption: no
webrtcAllowOrigin: "*"
webrtcLocalUDPAddress: ":8189"
webrtcLocalTCPAddress: ":8189"
webrtcIPsFromInterfaces: no
webrtcAdditionalHosts: ["$${PUBLIC_IP}"]
webrtcICEServers2: []

# API configuration
api: yes
apiAddress: ":9997"

# Path configuration (s0-s2 with YOLOv8 processing)
pathDefaults:
  source: publisher
  rtspTransport: tcp

paths:
  in_s0:
    source: publisher
    rtspTransport: tcp
  in_s1:
    source: publisher
    rtspTransport: tcp
  in_s2:
    source: publisher
    rtspTransport: tcp
  s0:
    source: rtsp://127.0.0.1:9500/ds-test
    rtspTransport: tcp
  s1:
    source: rtsp://127.0.0.1:9501/ds-test
    rtspTransport: tcp
  s2:
    source: rtsp://127.0.0.1:9502/ds-test
    rtspTransport: tcp
EOF

echo "[startup] Starting FRP server..."
docker rm -f frps >/dev/null 2>&1 || true
docker run -d \
  --name frps \
  --restart unless-stopped \
  --network host \
  -v /etc/frp/frps.ini:/etc/frp/frps.ini:ro \
  snowdreamtech/frps:0.51.3 \
  -c /etc/frp/frps.ini

echo "[startup] Starting MediaMTX container..."
docker rm -f mediamtx >/dev/null 2>&1 || true
docker run -d \
  --name mediamtx \
  --restart unless-stopped \
  --network host \
  -v /etc/mediamtx/config.yml:/mediamtx.yml:ro \
  bluenviron/mediamtx:latest

echo "[startup] Done. Logs: docker logs -f mediamtx"
