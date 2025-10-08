#!/usr/bin/env bash
set -euo pipefail

# End-to-end teardown + bringup + smoke test of a MediaMTX relay in GCP.
# Uses gcloud (no Terraform dependency) so it can run anywhere quickly.

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
ZONE="${ZONE:-asia-south1-c}"
INSTANCE="${INSTANCE:-mediamtx-relay-test}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-medium}"
PATH_REGEX="${PATH_REGEX:-^s([0-9]|[1-5][0-9]|6[0-3])$}"
FWR="${FWR:-mediamtx-allow-test}"

if [[ -z "$PROJECT" ]]; then
  echo "PROJECT is not set and no gcloud default project is configured." >&2
  echo "Set PROJECT=<gcp-project-id> or run: gcloud config set project <id>" >&2
  exit 1
fi

echo "[1/6] Teardown old instance + firewall (if any)"
gcloud -q compute instances delete "$INSTANCE" --zone "$ZONE" || true
gcloud -q compute firewall-rules delete "$FWR" || true

echo "[2/6] Create firewall rules"
gcloud compute firewall-rules create "$FWR" \
  --network=default --direction=INGRESS --action=ALLOW \
  --rules=tcp:8554,tcp:1935,tcp:8888,tcp:8889,tcp:9997,udp:8000-8001,udp:8189,udp:8890 \
  --target-tags=mediamtx-test >/dev/null

echo "[3/6] Render startup script from template"
TMP_STARTUP="$(mktemp)"
sed "s#\${path_regex}#$PATH_REGEX#g" "$(dirname "$0")/scripts/startup.sh" > "$TMP_STARTUP"

echo "[4/6] Create instance: $INSTANCE in $ZONE"
gcloud compute instances create "$INSTANCE" \
  --zone "$ZONE" \
  --machine-type "$MACHINE_TYPE" \
  --tags=mediamtx-test \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --metadata-from-file startup-script="$TMP_STARTUP" >/dev/null

echo "[5/6] Wait for services and obtain IP"
IP=$(gcloud compute instances describe "$INSTANCE" --zone "$ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "Relay IP: $IP"

echo "Polling RTSP OPTIONS on $IP:8554 ..."
for i in {1..30}; do
  if printf 'OPTIONS rtsp://%s:8554/s0 RTSP/1.0\r\nCSeq: 1\r\n\r\n' "$IP" | nc -w 2 "$IP" 8554 >/dev/null 2>&1; then
    break; fi; sleep 2; done

echo "[6/6] Publish a 5s test to rtsp://$IP:8554/s0"
# Try from local DS sender container if present; otherwise use host gst if available
CID=$(docker ps --filter ancestor=batch_streaming:latest --format '{{.ID}}' | head -n1 || true)
if [[ -n "$CID" ]]; then
  docker exec "$CID" bash -lc \
    "gst-launch-1.0 -q -e videotestsrc is-live=true num-buffers=150 ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bframes=0 bitrate=3000 ! h264parse config-interval=-1 ! rtspclientsink location=rtsp://$IP:8554/s0 protocols=tcp" || true
else
  if command -v gst-launch-1.0 >/dev/null 2>&1; then
    gst-launch-1.0 -q -e videotestsrc is-live=true num-buffers=150 ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bframes=0 bitrate=3000 ! h264parse config-interval=-1 ! rtspclientsink location=rtsp://$IP:8554/s0 protocols=tcp || true
  else
    echo "No GStreamer available locally and no sender container found; skipping publish test." >&2
  fi
fi

echo "Open WebRTC: http://$IP:8889/s0/ (should play during/after publish)"
echo "Logs on relay: gcloud compute ssh $INSTANCE --zone=$ZONE --command 'sudo docker logs --tail 100 mediamtx'"
