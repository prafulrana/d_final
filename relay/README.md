Relay Infrastructure (GCP)

This folder contains Infrastructure-as-Code to deploy a MediaMTX relay on Google Cloud (GCE) that accepts RTSP publishes (RECORD) and serves WebRTC/HLS/RTSP to viewers.

## Current Architecture (DeepStream + frpc Tunneling)

The relay is configured to support a DeepStream processing pipeline with 3 streams (s0-s2):

1. **Input**: Cameras publish to `in_s0`, `in_s1`, `in_s2` on the relay
2. **Processing**: DeepStream machines pull from `in_sX`, run inference (YOLOv8), and serve processed RTSP locally:
   - s0: `localhost:8554/ds-test` (YOLOv8 inference on in_s0)
   - s1: `localhost:8555/ds-test` (YOLOv8 inference on in_s1)
   - s2: `localhost:8556/ds-test` (YOLOv8 inference on in_s2)
3. **Tunneling**: frpc tunnels local RTSP to relay:
   - `localhost:8554` → `relay:9500`
   - `localhost:8555` → `relay:9501`
   - `localhost:8556` → `relay:9502`
4. **Relay Output**: MediaMTX pulls from `localhost:9500-9502` and serves as `s0-s2` via WebRTC/HLS/RTSP

**frpc Config**: `/root/d_final/frpc/frpc.ini`
**MediaMTX Config**: Relay pulls from `rtsp://127.0.0.1:950X/ds-test` for each sX path

## IMPORTANT: Static IP Configuration

The relay now uses a **static IP** (`google_compute_address`) to prevent IP changes on destroy/apply.

**Current Static IP**: `34.47.221.242`
**Static IP Resource**: `mediamtx-relay-ip` (managed by Terraform)

This means you can safely destroy and recreate the VM without changing the IP.

## Terraform Configuration

Defaults:
- Project: set via `-var project_id=...`
- Zone: `asia-south1-c` (override with `-var zone=...`)
- Instance name: `mediamtx-relay`
- Machine type: `e2-medium`
- Image: Ubuntu 24.04 LTS
- **Static IP**: `mediamtx-relay-ip` (already exists in GCP)

What it creates:
- 1x Static IP address (if not exists)
- 1x GCE VM with a startup script that:
  - Installs Docker
  - Writes `/etc/mediamtx/config.yml`
  - Writes `/etc/frp/frps.ini`
  - Runs MediaMTX with `--network host`, name `mediamtx`, auto-restart
  - Runs frps (FRP server) with `--network host`, name `frps`, auto-restart
- Firewall rule (target tag `mediamtx`) opening:
  - TCP: 7000 (frp), 8554 (RTSP), 1935 (RTMP), 8888 (HLS), 8889 (WebRTC HTTP), 9997 (API), 9500-9600 (frp tunnels)
  - UDP: 8000-8001 (RTSP RTP/RTCP), 8189 (WebRTC ICE), 8890 (SRT), 9500-9600 (frp tunnels)

## Deploy

### First Time Setup
1) Authenticate for Terraform:
   ```bash
   gcloud auth application-default login
   ```

2) From this folder:
   ```bash
   terraform init
   terraform apply -var project_id=<YOUR_GCP_PROJECT>
   ```
   Optional: `-var zone=us-central1-a` `-var instance_name=my-relay` `-var machine_type=e2-standard-2`

### Updating Configuration

**IMPORTANT**: The relay uses **immutable infrastructure**. Changes to `scripts/startup.sh` require destroying and recreating the VM.

```bash
cd relay/

# 1. Edit scripts/startup.sh (modify MediaMTX paths, frps settings, etc.)

# 2. Destroy and recreate (IP stays the same due to static IP)
export GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token)
terraform destroy -var project_id=fsp-api-1 -auto-approve
terraform apply -var project_id=fsp-api-1 -auto-approve

# 3. Verify IP didn't change
terraform output external_ip
# Should still be: 34.47.221.242

# 4. Get the new frps_token (changes on each apply)
terraform output -raw frps_token

# 5. Update local frpc config
# Edit /root/d_final/frpc/frpc.ini with new token
# Then restart frpc: pkill frpc && nohup frpc -c /root/d_final/frpc/frpc.ini > /var/log/frpc.log 2>&1 &
```

## Outputs

- `external_ip` – public IP of the relay (static: 34.47.221.242)
- `frps_token` – authentication token for frp tunnels (changes on each apply)

## Validate

RTSP (OPTIONS):
```bash
printf 'OPTIONS rtsp://34.47.221.242:8554/s0 RTSP/1.0\r\nCSeq: 1\r\n\r\n' | nc -w 3 34.47.221.242 8554
```

API:
```bash
curl http://34.47.221.242:9997/v3/paths/list
```

WebRTC:
- Input streams: http://34.47.221.242:8889/in_s0/, in_s1/, in_s2/
- Processed streams: http://34.47.221.242:8889/s0/, s1/, s2/

## Publish Test (from any host with GStreamer)

```bash
gst-launch-1.0 -e videotestsrc is-live=true ! videoconvert ! \
  x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bframes=0 ! \
  h264parse config-interval=-1 ! \
  rtspclientsink location=rtsp://34.47.221.242:8554/in_s0 protocols=tcp
```

## Monitor & Operate

SSH:
```bash
gcloud compute ssh mediamtx-relay --zone=asia-south1-c
```

Container logs:
```bash
sudo docker logs -f mediamtx
sudo docker logs -f frps
```

Service restart:
```bash
sudo docker restart mediamtx
sudo docker restart frps
```

Serial console:
```bash
gcloud compute instances get-serial-port-output mediamtx-relay --zone=asia-south1-c
```

API probe:
```bash
curl http://127.0.0.1:9997/v3/paths/list
```

Check frp tunnels:
```bash
docker logs frps | grep "new proxy"
# Should show: [s0_rtsp], [s1_rtsp], [s2_rtsp]
```

## Current MediaMTX Configuration

The startup script generates this config at `/etc/mediamtx/config.yml`:

```yaml
# Inputs (cameras publish here)
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

# Outputs (pull from frp-tunneled DeepStream)
  s0:
    source: rtsp://127.0.0.1:9500/ds-test
    rtspTransport: tcp
  s1:
    source: rtsp://127.0.0.1:9501/ds-test
    rtspTransport: tcp
  s2:
    source: rtsp://127.0.0.1:9502/ds-test
    rtspTransport: tcp
```

## Local DeepStream Setup (After Relay Deploy)

After deploying the relay, configure your DeepStream machine:

1. **Update source files** with relay IP (34.47.221.242):
   - `live_s0.c`, `live_s1.c`, `live_s2.c`
   - `s0_rtsp.py`
   - `loop_stream.sh`

2. **Update frpc config** (`frpc/frpc.ini`):
   ```ini
   [common]
   server_addr = 34.47.221.242
   server_port = 7000
   token = <frps_token from terraform output>
   ```

3. **Rebuild and start**:
   ```bash
   ./build.sh
   pkill frpc && nohup frpc -c /root/d_final/frpc/frpc.ini > /var/log/frpc.log 2>&1 &
   ./start.sh
   ```

4. **Verify frpc connected**:
   ```bash
   tail -10 /var/log/frpc.log
   # Should see: "proxy added: [s0_rtsp s1_rtsp s2_rtsp]"
   ```

## Troubleshooting

### Relay can't pull from 9500-9502 ("connection refused")
**Cause**: frpc not connected or DeepStream not running
**Fix**:
1. Check frpc logs on DeepStream machine: `tail -20 /var/log/frpc.log`
2. Verify DeepStream containers running: `docker ps | grep ds-s`
3. Check relay frps logs: `gcloud compute ssh mediamtx-relay --zone=asia-south1-c --command="docker logs frps --tail 20"`

### New frps_token after terraform apply
**Cause**: Terraform generates new random token on each apply
**Fix**: Update `frpc/frpc.ini` on DeepStream machine and restart frpc

### Stream works in in_sX but not sX
**Cause**: frpc tunnel not working
**Fix**: Restart frpc on DeepStream machine, check logs for "proxy added"

## Security Notes

This opens public ports. Lock down with source ranges on the firewall or enable authentication in `/etc/mediamtx/config.yml` (`authInternalUsers`) if required.

**Current setup**: No authentication, all paths publicly accessible.

## Multi-Zone / Different Zones

Override zone at apply time: `-var zone=us-central1-a`
Deploy multiple relays: use different `instance_name` (and optionally different zones).

## Changing Static IP

If you need to change the static IP:
```bash
# 1. Destroy existing static IP resource
gcloud compute addresses delete mediamtx-relay-ip --region=asia-south1

# 2. Apply with new address (or let Terraform create a new one)
terraform destroy -var project_id=fsp-api-1 -auto-approve
terraform apply -var project_id=fsp-api-1 -auto-approve

# 3. Update all 6 files on DeepStream machine (see AGENTS.md)
```
