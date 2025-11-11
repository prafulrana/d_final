# Repository Guidelines

This repo runs a single DeepStream container (ds-single) for YOLOv8 inference on live RTSP streams with dynamic source management (up to 36 concurrent streams).

## 🚨 HARDWARE CAPABILITY BASELINE

**This hardware (RTX 5080) has been tested running 168 concurrent DeepStream containers with YOLOWorld on vanilla DeepStream.**

**NEVER** claim anything is "too big", "massive", "heavy", or "resource intensive" for this hardware. If something uses high GPU/CPU, it's a **BUG or MISCONFIGURATION**, not a hardware limit.

**Models tested at scale:**
- YOLOWorld (largest YOLO variant) - 168 concurrent streams ✓
- YOLO12x (226MB ONNX) - Proven working at scale
- YOLOv8n (13MB ONNX) - Lightweight baseline

**Expected performance per stream (FP16 TensorRT + NVDEC decode + batching):**
- CPU: <10% per container (GPU-bound workload)
- GPU: 30-50% for YOLO12x @1280 (single stream with FP16)
- GPU: 45-75% only if FP32/CPU decode (misconfiguration)
- Any single stream using >90% GPU = BUG (wrong precision mode, CPU decode, no batching, or corrupted engine cache)

**Debug checklist for high GPU usage:**
1. Check `network-mode=2` (FP16) in config, not FP32
2. Verify `cudadec-memtype=0` (NVDEC GPU decode)
3. Confirm TensorRT `.engine` file exists and is valid
4. Enable batching: `batch-size=36` and `process-mode=1`
5. Check for NVENC/display overhead (`nvdsosd` + `nveglglessink`)
6. Delete and rebuild `.engine` files if switching models

## ⚠️ IMPORTANT: Container Management

**Use provided scripts for lifecycle management:**
- `./build.sh` - Build ds-single Docker image
- `./run.sh` - Start ds-single container
- `docker logs ds-single` - View logs
- `docker stop ds-single` - Stop container

## Architecture

### Current Setup (Single Container, Dynamic Streams)

**ds-single container:**
- Pulls from relay RTSP: `rtsp://34.47.221.242:8554/in_s{0..35}`
- Serves on local RTSP: `rtsp://localhost:9600/x{0..35}`
- HTTP API on port 5555 for stream control
- Supports up to 36 concurrent streams dynamically

**Key Characteristics:**
- Flask HTTP API runs immediately on container start
- Pipeline created lazily on first `/stream/restart` call (0→1 transition)
- Sources added/removed dynamically via HTTP API
- Single YOLOv8 inference engine processes all streams (batched)

### Data Flow
```
Cameras → Relay (in_s0, in_s1, ..., in_s35)
         ↓
HTTP POST /stream/restart {"id": N}
         ↓
DeepStream creates source N → YOLOv8 inference → Local RTSP (localhost:9600/xN)
         ↓
Relay pulls from localhost → Serves as sN (WebRTC/HLS/RTSP)
```

### Architecture Components

**HTTP Control API** (Flask, always running):
- `POST /stream/restart {"id": N}` - Start/restart stream N (creates pipeline on first call)
- `GET /stream/status` - Get active streams and pipeline status
- Listens on port 5555

**DeepStream Pipeline** (created on-demand):
- **nvstreammux**: Batches multiple streams (up to 36)
- **nvinfer**: YOLOv8 inference (single PGIE for all streams)
- **nvstreamdemux**: Separates streams back to individual outputs
- **Per-stream output chain**: nvvidconv → nvosd → encoder → rtppay → udpsink

**RTSP Server** (GstRtspServer):
- Port 9600
- Paths: `/x0`, `/x1`, `/x2`, ..., `/x35`
- RTSP factories created dynamically as streams are added

### Code Structure

**3-file Python architecture:**
```
config.py (26 lines)    - Configuration constants
pipeline.py (482 lines) - GStreamer pipeline and lifecycle
main.py (103 lines)     - Flask HTTP API
```

**Separation of concerns:**
- `config.py`: All constants (GPU_ID, ports, paths, dimensions)
- `pipeline.py`: All GStreamer code (pipeline creation, source management, callbacks)
- `main.py`: HTTP API only (Flask routes, lifecycle coordination)

**Lifecycle management:**
- Container starts → Flask runs immediately → Pipeline NOT created
- First `/stream/restart` call → Creates pipeline + first source
- Subsequent `/stream/restart` calls → Add sources to existing pipeline
- Pipeline persists until container stops (no 1→0 shutdown yet)

## Component Navigation

### Single Container System

**ds-single Container**:
- **Location**: `/root/d_final/`
- **Build**: `./build.sh`
- **Run**: `./run.sh`
- **Stop**: `docker stop ds-single`
- **Logs**: `docker logs ds-single`
- **Status**: `curl http://localhost:5555/stream/status`

**Configuration:**
- Command line URIs: Configured at container start via Dockerfile CMD
- Config file: `/config/bowling_yolo12n_batch.txt` (mounted at runtime)
- Model: `/models/` directory (bind-mounted from host)

## Critical Files

### Source Code (Python)
1. `config.py` - All configuration constants
2. `pipeline.py` - GStreamer pipeline and source management
3. `main.py` - Flask HTTP API for stream control

### Configuration Files
1. `config/bowling_yolo12n_batch.txt` - YOLOv8 inference config
2. `Dockerfile` - Container build definition with source URIs

### Build and Run Scripts
1. `./build.sh` - Builds ds-single:latest image
2. `./run.sh` - Starts ds-single container with volume mounts

## Quick Workflow

### Start Container
```bash
./build.sh         # Rebuild if Python code changed
./run.sh          # Start ds-single container
```

### Start First Stream (Creates Pipeline)
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"id": 0}' \
  http://localhost:5555/stream/restart
```

### Add More Streams
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"id": 1}' \
  http://localhost:5555/stream/restart

curl -X POST -H "Content-Type: application/json" \
  -d '{"id": 2}' \
  http://localhost:5555/stream/restart
```

### Check Status
```bash
curl http://localhost:5555/stream/status | jq
```

### View Logs
```bash
docker logs ds-single --tail 50
docker logs ds-single --follow  # Live tail
```

### Stop Container
```bash
docker stop ds-single
```

## TensorRT Engine Management

### How Engine Caching Works

**DeepStream auto-manages engines:**
- Config specifies: `model-engine-file=/models/bowling_yolo12n_1280_batch36.engine`
- First run: Builds engine from ONNX (3-10 minutes)
- Subsequent runs: Loads cached engine from `/models/` (instant)
- `/models/` is bind-mounted → engines persist across container restarts

### Standard Workflow: Adding New Model

```bash
# 1. Export ONNX (in training/)
cp training/path/to/best.onnx models/new_model.onnx

# 2. Update config
vim config/new_model_config.txt
# Update: onnx-file=/models/new_model.onnx
# Update: model-engine-file=/models/new_model_b1_gpu0_fp16.engine

# 3. Update config.py to point to new config
vim config.py
# Update: PGIE_CONFIG_FILE = "/config/new_model_config.txt"

# 4. Rebuild and restart
./build.sh
./run.sh

# 5. Wait for engine build (3-10 min on first run)
docker logs ds-single --follow | grep "PLAYING"

# 6. Subsequent restarts will use cached engine (instant)
```

### Only Delete Engines If
1. **Testing fresh build** - want to time build process
2. **Debugging engine corruption** - engine crashes on deserialize
3. **Disk cleanup** - removing old unused engines

```bash
# Delete specific engine
rm /root/d_final/models/old_model_b1_gpu0_fp16.engine

# Container will rebuild on next start
```

## Performance

Single container can handle 36 streams at ~30 FPS with YOLOv8 in portrait mode (720x1280).

GPU memory usage: ~390 MiB per active stream

## Recent Changes

### Major Refactor (2025-11-11)
1. **3-file architecture**: Split monolithic main.py (542 lines) into config.py (26), pipeline.py (482), main.py (103)
2. **Lazy pipeline creation**: Flask runs immediately, pipeline created on first stream request (0→1)
3. **Clean separation**: Config → Pipeline → API layers
4. **Single container**: Simplified from multi-container to single ds-single with dynamic sources

### Benefits
- **Faster startup**: Flask API available immediately (no pipeline overhead)
- **Cleaner code**: 80% reduction in main.py complexity (542→103 lines)
- **Better separation**: Config, pipeline, and API fully decoupled
- **Dynamic scaling**: Add/remove streams via API without container restart

### Component Separation
- **Production system**: `/root/d_final/` - ds-single container with HTTP API
- **Training system**: `/root/d_final/training/` - model training and ONNX export

## Expectations for AI Agents

1. **Use HTTP API for stream control**: POST /stream/restart and GET /stream/status
2. **Check status before debugging**: Run `curl http://localhost:5555/stream/status` first
3. **Pipeline is lazy**: Not created until first stream requested
4. **Container is stateless**: Stop and restart with `./run.sh` to reset
5. **Code is modular**: Edit config.py for constants, pipeline.py for GStreamer, main.py for API
6. **Don't manually manage engines**: DeepStream auto-caches in `/models/`, persists across restarts
7. **Training separation**: Production code in `/root/d_final/`, training in `/root/d_final/training/`
8. **ALWAYS use TCP for RTSP probing**: Use `ffprobe -rtsp_transport tcp rtsp://...` NOT plain `ffprobe rtsp://...` (UDP times out)

## Training Workflow

### Separation of Concerns
- **Production**: `/root/d_final/` - deployed code (config.py, pipeline.py, main.py)
- **Training**: `/root/d_final/training/` - model training, ONNX export, experiments
- **Models**: `/root/d_final/models/` - production ONNX files and TensorRT engines

### Adding New Models
```bash
# 1. Train in /root/d_final/training/
cd /root/d_final/training
python3 train_bowling.py --imgsz 1280 --batch 8 --epochs 150

# 2. Export ONNX with FIXED dimensions (dynamic=False)
docker run --rm --gpus all -v /root/d_final/training:/data \
  ultralytics/ultralytics:latest yolo export \
  model=/data/runs/train/weights/best.pt \
  format=onnx imgsz=1280 dynamic=False simplify=True

# 3. Copy to production models/
cp training/runs/train/weights/best.onnx models/new_model.onnx

# 4. Update config file
cp config/bowling_yolo12n_batch.txt config/new_model_batch.txt
vim config/new_model_batch.txt
# Update: onnx-file=/models/new_model.onnx
# Update: model-engine-file=/models/new_model_b1_gpu0_fp16.engine

# 5. Update config.py
vim config.py
# Update: PGIE_CONFIG_FILE = "/config/new_model_batch.txt"

# 6. Rebuild and restart - engine builds automatically
./build.sh
./run.sh

# NO need to delete engines - DeepStream auto-rebuilds when missing
```

### Current Model
- **All streams**: YOLO12n Bowling @ 1280x1280, batch-size=36, portrait mode (720x1280)
