# Standards and Workflow

## Quick Reference

### Management Commands
- **`./build.sh`**: Rebuild ds-single Docker image
- **`./run.sh`**: Start ds-single container
- **`docker logs ds-single`**: View container logs
- **`docker stop ds-single`**: Stop container
- **`curl http://localhost:5555/stream/status`**: Check status

### Component Locations
- **Source code**: `config.py`, `pipeline.py`, `main.py`
- **Inference config**: `config/bowling_yolo12n_batch.txt`
- **Models**: `models/` (ONNX + TensorRT engines)

### Quick Commands
```bash
./build.sh                       # Rebuild image
./run.sh                        # Start container
curl http://localhost:5555/stream/status | jq  # Check status
docker logs ds-single --tail 50  # View logs
```

## Architecture Standards

### 3-File Python Structure

**config.py** (26 lines):
- All configuration constants only
- No logic, no imports except standard library
- Easy to read and modify

**pipeline.py** (482 lines):
- All GStreamer code
- Pipeline creation/destruction
- Source management
- All callbacks
- No Flask, no HTTP logic

**main.py** (103 lines):
- Flask HTTP API only
- Lifecycle coordination (0→1 creates pipeline)
- No GStreamer code

**Separation rules:**
- Config changes → edit `config.py` only
- Pipeline changes → edit `pipeline.py` only
- API changes → edit `main.py` only

## HTTP API Standards

### Endpoints

**POST /stream/restart**:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"id": 0}' \
  http://localhost:5555/stream/restart
```
- Creates pipeline on first call (id 0)
- Adds/restarts sources on subsequent calls
- Returns: `{"status": "ok", "id": N}` or error

**GET /stream/status**:
```bash
curl http://localhost:5555/stream/status
```
- Returns pipeline status and active streams
- Shows configured sources vs active streams
- Indicates if pipeline created or not

## Deployment Workflow

### After Code Changes

```bash
# 1. Edit Python files
vim config.py      # Or pipeline.py or main.py

# 2. Rebuild Docker image
./build.sh

# 3. Restart container
./run.sh

# 4. Check logs
docker logs ds-single --tail 50
```

**Always rebuild after code changes** - Docker uses cached image.

### After Config Changes

```bash
# 1. Edit inference config
vim config/bowling_yolo12n_batch.txt

# 2. Update config.py if needed
vim config.py
# Update: PGIE_CONFIG_FILE = "/config/new_config.txt"

# 3. Rebuild and restart
./build.sh
./run.sh
```

### Changing Models

```bash
# 1. Export ONNX (in training/)
cp training/runs/train/weights/best.onnx models/new_model.onnx

# 2. Create inference config
cp config/bowling_yolo12n_batch.txt config/new_model_batch.txt
vim config/new_model_batch.txt
# Update: onnx-file=/models/new_model.onnx
# Update: model-engine-file=/models/new_model_b1_gpu0_fp16.engine

# 3. Update config.py
vim config.py
# Update: PGIE_CONFIG_FILE = "/config/new_model_batch.txt"

# 4. Rebuild and restart (engine builds automatically)
./build.sh
./run.sh

# 5. Wait for engine build (3-10 min on first run)
docker logs ds-single --follow | grep "PLAYING"

# 6. Subsequent restarts use cached engine (instant)
```

## Performance Standards

### 30 FPS is Sacred

Maintain these settings in `config.py`:
- `MUXER_BATCH_TIMEOUT_USEC = 33000` (33ms for 30 FPS)
- `ENCODER_IDR_INTERVAL = 30` (IDR frame every 30 frames)

### Batch Size Consistency

All config locations must match:
- `config.py`: `MAX_NUM_SOURCES = 36`
- Inference config: `batch-size=36`
- TensorRT engine: Built for batch-size=36

Changing batch size requires:
1. Edit `config.py` MAX_NUM_SOURCES
2. Edit inference config batch-size
3. Delete cached `.engine` files
4. Rebuild and restart

## TensorRT Engine Management

### How Engine Caching Works

**DeepStream auto-manages engines:**
- Config specifies: `model-engine-file=/models/xxx_batch36.engine`
- First run: Checks `/models/`, not found → builds from ONNX (3-10 min)
- Saves to: `/models/xxx_batch36.engine` (bind-mounted, persists)
- Second run: Loads from `/models/xxx_batch36.engine` (instant)

**No manual copying needed** - engines saved directly to bind-mounted `/models/`

### Standard Operating Procedure: Adding New Model

**ALWAYS follow this sequence:**

```bash
# 1. Export ONNX with FIXED dimensions (dynamic=False)
cd /root/d_final/training
docker run --rm --gpus all -v $(pwd):/data \
  ultralytics/ultralytics:latest yolo export \
  model=/data/runs/train/weights/best.pt \
  format=onnx imgsz=1280 dynamic=False simplify=True

# 2. Copy to production models/
cp training/runs/train/weights/best.onnx models/new_model.onnx

# 3. Create/update inference config
cp config/bowling_yolo12n_batch.txt config/new_model_batch.txt
vim config/new_model_batch.txt
# Update: onnx-file=/models/new_model.onnx
# Update: model-engine-file=/models/new_model_b1_gpu0_fp16.engine
# Update: num-detected-classes=X
# Update: labelfile-path=/models/new_labels.txt

# 4. Update config.py
vim config.py
# Update: PGIE_CONFIG_FILE = "/config/new_model_batch.txt"

# 5. Rebuild and restart
./build.sh
./run.sh

# 6. Wait for "Pipeline ready" (3-10 min on first run)
docker logs ds-single --follow | grep "Pipeline ready"

# 7. Test first stream
curl -X POST -H "Content-Type: application/json" \
  -d '{"id": 0}' \
  http://localhost:5555/stream/restart

# 8. Verify cache on second restart (should be instant)
docker stop ds-single
./run.sh
# Should see "deserialized trt engine from :/models/..." in logs
```

### When to Delete Engines

**ONLY delete engines if:**
1. **Testing fresh build** - benchmarking build time
2. **Engine corrupted** - crashes with deserialize errors
3. **Disk cleanup** - removing old unused engines
4. **Config changed** - batch size, precision, input size

```bash
# Delete specific engine
rm /root/d_final/models/old_model_b1_gpu0_fp16.engine

# Or delete all engines
rm /root/d_final/models/*.engine

# Container will rebuild on next start
./run.sh
```

### How to Verify Caching Works

**Good (using cache)**:
```bash
docker logs ds-single 2>&1 | grep -i "deserialized trt engine"
```
Output should show:
```
deserialized trt engine from :/models/xxx_b1_gpu0_fp16.engine
Use deserialized engine model: /models/xxx_b1_gpu0_fp16.engine
```
Time to "Pipeline ready": <10 seconds

**Bad (rebuilding)**:
```bash
docker logs ds-single 2>&1 | grep -i "building"
```
Output shows:
```
deserialize engine from file :/models/xxx.engine failed
Building the TensorRT Engine
```
Time to "Pipeline ready": 3-10 minutes

## Debugging Workflow

### Quick Health Check
```bash
# 1. Check container status
docker ps | grep ds-single

# 2. Check API status
curl http://localhost:5555/stream/status | jq

# 3. Check logs
docker logs ds-single --tail 50

# 4. Check for errors
docker logs ds-single 2>&1 | grep -i error
```

### Stream Not Working
```bash
# 1. Check container is running
docker ps | grep ds-single

# 2. Check if pipeline created
curl http://localhost:5555/stream/status | jq .pipeline

# 3. Try restarting stream
curl -X POST -H "Content-Type: application/json" \
  -d '{"id": 0}' \
  http://localhost:5555/stream/restart

# 4. Check detailed logs
docker logs ds-single --follow
```

### Pipeline Won't Start
```bash
# 1. Check for engine build errors
docker logs ds-single 2>&1 | grep -i "error\|fail"

# 2. Check ONNX file exists
docker exec ds-single ls -lh /models/*.onnx

# 3. Check config file exists
docker exec ds-single ls -lh /config/*.txt

# 4. Try fresh restart
docker stop ds-single
./run.sh
```

## Git Practices

### Committing Changes

```bash
# After code changes
git add config.py pipeline.py main.py
git commit -m "refactor: description of changes"

# After config changes
git add config/bowling_yolo12n_batch.txt
git commit -m "config: update model parameters"

# After model changes
git add models/new_model.onnx models/new_labels.txt
git commit -m "feat: add new model"
```

### What to Commit
- **DO commit**: Python source, configs, ONNX models, labels
- **DON'T commit**: TensorRT `.engine` files (machine-specific)

### .gitignore
```
models/*.engine
*.pyc
__pycache__/
training/runs/
training/datasets/
.terraform/
terraform.tfstate*
```

## Viewing Streams

### RTSP Streams (Local)
```bash
# View processed stream locally
ffplay rtsp://127.0.0.1:9600/x0
ffplay rtsp://127.0.0.1:9600/x1
ffplay rtsp://127.0.0.1:9600/x2
```

### Via Relay (if configured)
- Processed streams: `http://RELAY_IP:8889/s0/`, `s1/`, `s2/`
- Input streams: `http://RELAY_IP:8889/in_s0/`, `in_s1/`, `in_s2/`

## Performance Monitoring

### Check FPS
```bash
docker logs ds-single | grep "**PERF"
# Should see ~30 FPS steady-state
```

### Check GPU Usage
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# Each active stream: ~390 MiB GPU memory
```

### Check CPU Usage
```bash
docker stats --no-stream ds-single
# Should be 4-5% CPU per stream at 30 FPS
```

## Common Issues

### Changes Don't Apply After Edit
**Cause**: Forgot to rebuild Docker image
**Fix**: `./build.sh` before `./run.sh`

### Pipeline Won't Create
**Cause**: Error in Python code or missing dependencies
**Fix**: Check logs with `docker logs ds-single --tail 100`

### Stream Stuck "Resetting source"
**Cause**: Can't pull from relay (wrong IP, relay down, network issue)
**Fix**: Check source URI in Dockerfile CMD, verify relay is running

### TensorRT Build Fails
**Cause**: Invalid ONNX, wrong precision, or insufficient GPU memory
**Fix**:
1. Verify ONNX with `onnxsim models/xxx.onnx models/xxx_simplified.onnx`
2. Check GPU memory with `nvidia-smi`
3. Try reducing batch size

### Container Won't Start
**Cause**: Port conflict or volume mount issue
**Fix**:
1. Check if port 5555 or 9600 in use: `netstat -tlnp | grep -E "5555|9600"`
2. Verify volume mounts exist: `ls -l /root/d_final/config/ /root/d_final/models/`

## Code Style

### Python
- Use f-strings for formatting (not % or .format())
- Keep functions focused (one responsibility)
- Document with docstrings
- No global state outside module globals

### Config Files
- Use consistent naming: `model_name_resolution_batch.txt`
- Always specify full paths (`/models/`, `/config/`)
- Keep batch-size consistent with config.py

### Logging
- Use `print()` for normal flow (captured by Docker logs)
- Use `sys.stderr.write()` for errors
- Include context in messages: `print(f"✓ Created source {source_id}")`

## Helper Scripts (Optional)

If `infra/restart_stream.sh` exists:
```bash
./infra/restart_stream.sh 0  # Restart stream 0
./infra/restart_stream.sh 1  # Restart stream 1
```

Otherwise use curl directly:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"id": 0}' \
  http://localhost:5555/stream/restart
```
