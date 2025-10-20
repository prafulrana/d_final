# Standards and Workflow

## Quick Reference

### Management Scripts
- **`./system`**: Full system (frpc + DeepStream) - `start|stop|restart|status`
- **`./ds`**: DeepStream containers only - `build|start|stop|restart|status`
- **`./relay`**: Remote relay - `restart|status`
- **`./build.sh`**: Rebuild Docker images

### Component Locations
- **DeepStream source**: `live_stream.c` (single parameterized binary)
- **FRP config**: `.scripts/frpc/frpc.ini`
- **Inference config**: `config/config_infer_yolov8.txt`
- **Utilities**: `.scripts/check.sh`, `.scripts/debug.sh`

### Quick Commands
```bash
./system start      # Start everything (frpc + containers)
./system status     # Health check
./ds restart        # Restart containers only
./relay status      # Check remote relay
```

## Architecture Constraints

### Relay IP Changes
When the relay IP changes (after terraform destroy/apply), you MUST update these 2 files:
1. `live_stream.c` (line 96: `snprintf(input_uri, ...)`)
2. `.scripts/frpc/frpc.ini` (line 2: `server_addr`, line 4: `token`)

Then rebuild and restart:
```bash
./build.sh
./system restart    # Restarts frpc + all containers
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
./system restart    # Restarts frpc + all containers together
```

**Check it worked**:
```bash
./system status     # Full health check
# Or check logs manually:
tail -10 /var/log/frpc.log
# Should see: "proxy added: [s0_rtsp s1_rtsp s2_rtsp]"
```

## DeepStream Config Changes

### Editing Source Code
After editing `live_stream.c`, you MUST rebuild:
```bash
./build.sh      # Rebuilds ds-s1:latest with new binary
./ds restart    # Restart all DeepStream containers
```

Docker uses cached binaries - your edits won't apply until rebuild.

### Changing Inference Model
```bash
# Edit live_stream.c line 150 to use different config
sed -i 's|config_infer_yolov8.txt|config_infer_new.txt|' live_stream.c

# Rebuild and restart
./build.sh
rm models/*.engine  # Clear old TensorRT engines
./ds restart
```

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
./ds restart    # All containers will try to build - wait for completion
```

## Debugging Workflow

### Quick Health Check
```bash
./system status    # Full system health check
./.scripts/debug.sh # Detailed diagnostics
```

### Stream Not Working
```bash
# 1. Check container is running
./ds status

# 2. Check container logs
docker logs ds-s0 --tail 50

# 3. Check frpc tunnel
tail -20 /var/log/frpc.log

# 4. Check relay
./relay status
```

### Don't Modify Working Streams
If s0 and s1 work but s2 doesn't:
- **DO**: Debug s2
- **DON'T**: Change s0 or s1

They have identical architecture (same binary, different stream ID). If one works, the others should too.

## Git Practices

### Committing Changes
```bash
# After relay IP change, commit all updated files
git add live_stream.c .scripts/frpc/frpc.ini
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

