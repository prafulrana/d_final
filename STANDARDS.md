# Standards and Workflow

## Architecture Constraints

### Relay IP Changes
When the relay IP changes (after terraform destroy/apply), you MUST update these 3 files:
1. `live_stream.c` (line 97: `snprintf(input_uri, ...)`)
2. `frpc/frpc.ini` (line 2: `server_addr`, line 4: `token`)
3. `publisher/loop_stream.sh` (line 7: `rtspclientsink location`)

Then rebuild and restart:
```bash
./build.sh
pkill frpc && nohup frpc -c /root/d_final/frpc/frpc.ini > /var/log/frpc.log 2>&1 &
./start.sh
```

### Relay Configuration (Terraform)
The relay is **immutable infrastructure**. To change MediaMTX config:
```bash
cd relay/
# 1. Edit relay/scripts/startup.sh
# 2. Destroy and recreate
export GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token)
terraform destroy -var project_id=fsp-api-1 -auto-approve
terraform apply -var project_id=fsp-api-1 -auto-approve
# 3. Get new IP and token
terraform output external_ip
terraform output -raw frps_token
# 4. Update all 3 files with new IP/token
# 5. Rebuild and restart (see above)
```

### frpc (Local FRP Client)
**Always restart after config changes**:
```bash
pkill frpc
nohup frpc -c /root/d_final/frpc/frpc.ini > /var/log/frpc.log 2>&1 &
```

**Check it worked**:
```bash
tail -10 /var/log/frpc.log
# Should see: "proxy added: [s0_rtsp s1_rtsp s2_rtsp]"
```

## DeepStream Config Changes

### Editing Source Code
After editing `live_stream.c`, you MUST rebuild:
```bash
./build.sh    # Rebuilds ds-s1:latest with new binary
./stop.sh     # Stop all containers
./start.sh    # Restart all containers (includes publisher)
```

Docker uses cached binaries - your edits won't apply until rebuild.

### Changing Inference Model
```bash
# Edit live_stream.c line 165 to use different config
sed -i 's|config_infer_yolov8.txt|config_infer_new.txt|' live_stream.c

# Rebuild and restart
./build.sh
rm models/*.engine  # Clear old TensorRT engines
./start.sh
```

### OSD/Overlay Changes
Edit `config/config_osd.txt` then restart:
```bash
./start.sh
```

OSD config is read at startup, no rebuild needed.

## Performance Standards

### 30 FPS is Sacred
Maintain these settings in `live_stream.c`:
- `batched-push-timeout=33333` (nvstreammux, line 159)
- `iframeinterval=30` (encoder, line 181)
- `live-source=0` (nvstreammux, line 161)

### Batch Size Consistency
All 3 streams use batch-size=1:
- nvstreammux: `batch-size=1` (live_stream.c line 158)
- Inference config: `batch-size=1` (config/config_infer_yolov8.txt line 14)
- TensorRT engine: Built for batch-size=1

Changing batch size requires:
1. Edit live_stream.c line 158
2. Edit `config/config_infer_yolov8.txt` line 14
3. Delete `models/*.engine`
4. Rebuild and restart

## TensorRT Engine Management

### First Build (No Cached Engine)
**IMPORTANT**: Avoid race conditions by building engines serially:
```bash
# Stop all containers
docker stop ds-s0 ds-s1 ds-s2

# Start ONLY s0 to build engine
docker start ds-s0

# Wait ~3-5 minutes for engine build
# Watch for "deserialized trt engine" in logs

# Once complete, copy engine to host and restart all
docker exec ds-s0 cp /app/model_b1_gpu0_fp16.engine /models/yolov8n_b1_gpu0_fp16.engine
./start.sh
```

### Rebuilding Engines
```bash
rm models/*.engine
./start.sh    # All containers will try to build - wait for completion
```

## Debugging Workflow

### Stream Not Working
```bash
# 1. Check container is running
docker ps | grep ds-s

# 2. Check container logs
docker logs ds-s0 --tail 50

# 3. Check local RTSP server
ffprobe rtsp://127.0.0.1:8554/ds-test

# 4. Check frpc tunnel
tail -20 /var/log/frpc.log

# 5. Check relay
gcloud compute ssh mediamtx-relay --zone=asia-south1-c \
  --command="docker logs mediamtx --tail 20"
```

### Don't Modify Working Streams
If s0 and s1 work but s2 doesn't:
- **DO**: Debug s2
- **DON'T**: Change s0 or s1

They have identical architecture (same binary, different stream ID). If one works, the others should too.

## Git Practices

### Committing Changes
```bash
# After relay IP change, commit all 3 updated files
git add live_stream.c frpc/frpc.ini publisher/loop_stream.sh
git commit -m "config: update relay IP to $(cd relay && terraform output -raw external_ip)"
```

### What to Commit
- **DO commit**: Config files, source code (.c), docs
- **DON'T commit**: TensorRT .engine files (machine-specific)

### .gitignore
```
models/*.engine
*.pyc
__pycache__/
.terraform/
terraform.tfstate*
```

## Viewing Streams

### Processed Streams (After YOLOv8)
- s0: http://34.47.221.242:8889/s0/
- s1: http://34.47.221.242:8889/s1/
- s2: http://34.47.221.242:8889/s2/

### Input Streams (Before Processing)
- in_s0: http://34.47.221.242:8889/in_s0/
- in_s1: http://34.47.221.242:8889/in_s1/
- in_s2: http://34.47.221.242:8889/in_s2/

### Local RTSP (For Testing)
```bash
ffplay rtsp://127.0.0.1:8554/ds-test  # s0
ffplay rtsp://127.0.0.1:8555/ds-test  # s1
ffplay rtsp://127.0.0.1:8556/ds-test  # s2
```

## Performance Monitoring

### Check FPS
```bash
docker logs ds-s0 | grep "**PERF"
docker logs ds-s1 | grep "**PERF"
docker logs ds-s2 | grep "**PERF"
```

Should see ~30 FPS steady-state.

### Check GPU Usage
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Each container uses ~390 MiB GPU memory.

### Check CPU Usage
```bash
docker stats --no-stream
```

Each container should use 4-5% CPU at 30 FPS.

## Common Issues

### "Resetting source" Loop
**Cause**: Container can't pull from relay (wrong IP, relay down, network issue)
**Fix**: Check relay IP in live_stream.c line 97, verify relay is running

### "Connection refused" on 9500-9502
**Cause**: frpc not running or not connected
**Fix**: Restart frpc and check logs

### Stream Works in in_sX But Not sX
**Cause**: frpc tunnel not working or relay can't pull from tunnel
**Fix**:
1. Verify frpc logs show "proxy added"
2. Check relay logs for "ready: 1 track"
3. Restart ds-sX container if stuck

### Changes to live_stream.c Don't Apply
**Cause**: Forgot to rebuild Docker image
**Fix**: `./build.sh` before `./start.sh`

### TensorRT Build Hangs/Fails
**Cause**: Multiple containers trying to build same engine file simultaneously
**Fix**: Stop all containers, let one build first, then restart all

## Publisher (Test Video Streaming)

### Start Publisher
```bash
cd publisher/
./start.sh    # Streams test video to relay's in_s2
```

### Stop Publisher
```bash
docker stop publisher
```

### Change Test Video
Replace `publisher/bowling_bottom_right.mp4` with your video file, then:
```bash
cd publisher/
docker build -t publisher:latest .
./start.sh
```

### Publisher Architecture
- Runs in separate container
- Loops video file continuously
- Publishes to `rtsp://34.47.221.242:8554/in_s2` via TCP
- Good for testing s2 without needing actual camera
