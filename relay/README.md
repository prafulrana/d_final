Relay Infrastructure (GCP)

This folder contains Infrastructure-as-Code to deploy a MediaMTX relay on Google Cloud (GCE) that accepts RTSP publishes (RECORD) and serves WebRTC/HLS/RTSP to viewers.

Defaults
- Project: set via `-var project_id=...`
- Zone: `asia-south1-c` (override with `-var zone=...`)
- Instance name: `mediamtx-relay`
- Machine type: `e2-medium`
- Image: Ubuntu 24.04 LTS

What it creates
- 1x GCE VM with a startup script that:
  - Installs Docker
  - Writes `/etc/mediamtx/config.yml`
  - Runs MediaMTX with `--network host`, name `mediamtx`, auto-restart
- Firewall rule (target tag `mediamtx`) opening:
  - TCP: 8554 (RTSP), 1935 (RTMP), 8888 (HLS), 8889 (WebRTC HTTP), 9997 (API)
  - UDP: 8000-8001 (RTSP RTP/RTCP), 8189 (WebRTC ICE), 8890 (SRT)

Paths policy
- By default, the relay accepts publishes to `s0..s63` with a single regex.
- Change the allowed path regex with `-var path_regex=...` or switch to `paths: { all: {} }` in the startup script template.

Deploy
1) Authenticate for Terraform (one option):
   - `gcloud auth application-default login`
2) From this folder:
   - `terraform init`
   - `terraform apply -var project_id=<YOUR_GCP_PROJECT>`
     - Optional: `-var zone=us-central1-a` `-var instance_name=my-relay` `-var machine_type=e2-standard-2`

Outputs
- `external_ip` – public IP of the relay

Validate
- RTSP (OPTIONS):
  - `printf 'OPTIONS rtsp://<external_ip>:8554/s0 RTSP/1.0\r\nCSeq: 1\r\n\r\n' | nc -w 3 <external_ip> 8554`
- API (if enabled):
  - `curl http://<external_ip>:9997/v3/paths/list`
- WebRTC:
  - `http://<external_ip>:8889/s0/` (shows a player when a publisher is active)

Publish test (from any host with GStreamer)
- `gst-launch-1.0 -e videotestsrc is-live=true ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bframes=0 ! h264parse config-interval=-1 ! rtspclientsink location=rtsp://<external_ip>:8554/s0 protocols=tcp`

Monitor & Operate
- SSH: `gcloud compute ssh mediamtx-relay --zone=${var.zone}`
- Container logs: `sudo docker logs -f mediamtx`
- Service restart: `sudo docker restart mediamtx`
- Serial console: `gcloud compute instances get-serial-port-output mediamtx-relay --zone=${var.zone}`
- API probe: `curl http://127.0.0.1:9997/v3/paths/list`

Multi‑zone / Different zones
- Override zone at apply time: `-var zone=us-central1-a`
- Deploy multiple relays: use different `instance_name` (and optionally different zones).

Security notes
- This opens public ports. Lock down with source ranges on the firewall or enable authentication in `/etc/mediamtx/config.yml` (`authInternalUsers`) if required.

