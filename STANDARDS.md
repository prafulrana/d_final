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
# 1. Create new inference config
cp config/config_infer_yolov8.txt config/config_infer_new.txt
vim config/config_infer_new.txt
# Update: onnx-file=/models/new_model.onnx
# Update: model-engine-file=/models/new_model_b1_gpu0_fp16.engine

# 2. Edit live_stream.c line 150 to use different config
sed -i 's|config_infer_yolov8.txt|config_infer_new.txt|' live_stream.c

# 3. Rebuild and restart (no engine deletion needed)
./build.sh
./ds restart

# DeepStream auto-builds new engine, old engines remain cached
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

### Critical Understanding: DeepStream Saves to /app/, NOT /models/

**The Problem**:
- Config says: `model-engine-file=/models/xxx.engine` (where DeepStream LOADS from)
- DeepStream saves to: `/app/model_b1_gpu0_fp16.engine` (hardcoded location)
- `/app/` is ephemeral (lost on restart), `/models/` is bind-mounted (persistent)
- Without manual intervention, engines rebuild on every container restart

**The Solution**: Use `.scripts/cache_engine.sh` to copy engines from `/app/` → `/models/`

### Standard Operating Procedure: Adding New Model

**ALWAYS follow this sequence**:

```bash
# 1. Export ONNX using DeepStream-Yolo method (ensures proper output layer)
cd /root/d_final/training
./scripts/export_deepstream.sh path/to/best.pt 640
# This uses DeepStream-Yolo's export_yoloV8.py with Ultralytics container
# Adds DeepStreamOutput layer for compatibility with custom parser

# 2. Copy ONNX to production models/
cp training/path/to/best.pt.onnx models/new_model.onnx

# 3. Create/update config file
cp config/config_infer_yolov8.txt config/config_infer_new.txt
vim config/config_infer_new.txt
# Update: onnx-file=/models/new_model.onnx
# Update: model-engine-file=/models/new_model_b1_gpu0_fp16.engine
# Update: num-detected-classes=X
# Update: labelfile-path=/models/new_labels.txt

# 4. Update ds script or live_stream.c to use new config
# For s2: Edit ds script line 22 (config path argument)
# Or: Edit live_stream.c default config logic

# 5. Rebuild Docker image (if live_stream.c changed)
./build.sh

# 6. Start containers - engines build in /app/ (3-10 min)
./ds restart

# 7. Wait for "Pipeline set to PLAYING"
docker logs ds-s2 --follow | grep "PLAYING"
# Press Ctrl+C when you see: "Pipeline set to PLAYING"

# 8. CRITICAL: Copy engines to persistent cache
./.scripts/cache_engine.sh copy

# 9. Verify cache with verify command
./.scripts/cache_engine.sh verify
# Should show ✓ for all configs pointing to existing cached engines

# 10. List engines to confirm cache status
./.scripts/cache_engine.sh list
# Should show engines in both /app/ (containers) and /models/ (host)

# 11. Test cache reuse - restart should be instant (no rebuild)
./ds restart
docker logs ds-s2 2>&1 | grep "deserialized trt engine"
# Should see: "deserialized trt engine from :/models/new_model_b1_gpu0_fp16.engine"

# 12. Verify no rebuild happened
docker logs ds-s2 2>&1 | grep "Building the TensorRT Engine"
# Should be EMPTY (no output = good, used cache)
```

### cache_engine.sh Commands

```bash
# Copy engines from /app/ to /models/ with correct names
./.scripts/cache_engine.sh copy

# List engines in containers + /models/
./.scripts/cache_engine.sh list

# Verify configs point to existing cached engines
./.scripts/cache_engine.sh verify

# Delete cached engines (force rebuild)
./.scripts/cache_engine.sh clean
```

### When to Run cache_engine.sh copy

**ALWAYS run after**:
1. First deployment of new model
2. Any ONNX model change (forces rebuild)
3. Moving to different GPU (different CUDA compute capability)
4. Config changes that affect engine (batch size, precision, etc.)

**Skip if**:
- Just restarting containers with existing cached engines
- No model or config changes

### How to Verify Caching Works

**Good (using cache)**:
```bash
docker logs ds-s2 2>&1 | grep -i engine | head -5
```
Output should show:
```
deserialized trt engine from :/models/xxx_b1_gpu0_fp16.engine
Use deserialized engine model: /models/xxx_b1_gpu0_fp16.engine
```
Time to PLAYING: <10 seconds

**Bad (rebuilding)**:
```
deserialize engine from file :/models/xxx.engine failed
Building the TensorRT Engine
```
Time to PLAYING: 3-10 minutes

### Only Delete Engines If

1. **Testing fresh build** - benchmarking build time
2. **Engine corrupted** - crashes with deserialize errors
3. **Disk cleanup** - removing old unused engines

```bash
# Delete all cached engines
./.scripts/cache_engine.sh clean

# Or delete specific engine
rm /root/d_final/models/old_model_b1_gpu0_fp16.engine
```

After deletion, containers will rebuild engines in `/app/` on next start. **Remember to run `./.scripts/cache_engine.sh copy` after rebuild!**

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

### FFmpeg Publishing Works But DeepStream Crashes with "NvBufSurfTransform failed"
**Cause**: iPhone HDR video (Dolby Vision, bt2020 color space) incompatible with DeepStream
**Symptoms**:
- Pipeline sets to PLAYING, receives pad, links succeed
- Then crashes with: `nvbufsurftransform.cpp:4253: => Transformation Failed -1`
- No output, no PERF stats, container exits
- Works perfectly with Larix but not FFmpeg from iPhone video file

**Fix**: Convert iPhone HDR video to SDR (bt709) before streaming:

```bash
# Step 1: Convert iPhone video to clean SDR file (one-time, slow but high quality)
ffmpeg -i your_iphone_video.MOV \
  -vf "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709,format=yuv420p,fps=30,scale=1920:1080" \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -c:v libx264 -profile:v baseline -bf 0 -g 30 -preset slow \
  -c:a aac -b:a 128k -ar 48000 \
  -movflags +faststart \
  clean_video.mp4

# Step 2: Stream the clean file (fast, zero re-encoding)
ffmpeg -re -stream_loop -1 -i clean_video.mp4 \
  -c copy \
  -f rtsp -rtsp_transport tcp \
  rtsp://34.47.221.242:8554/in_s0
```

**Why this happens**:
- iPhone 15 Pro Max (and similar) record in Dolby Vision HDR
- Source video is 10-bit yuv420p10le with bt2020 color space
- DeepStream expects 8-bit SDR (bt709) video
- NvBufSurfTransform can't convert bt2020 → NV12 for inference
- The conversion must happen in FFmpeg, not in DeepStream

**Larix works because**: Mobile apps encode to baseline H264 with bt709 (standard SDR) by default

