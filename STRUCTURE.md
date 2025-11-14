# Project Structure

```
d_final/
├── AGENTS.md                    # Architecture + operational guide
├── STANDARDS.md                 # Coding/config standards
├── STRUCTURE.md                 # This file
│
├── build.sh                    # Docker image builder (ds-single:latest)
├── run.sh                      # Run ds-single container
│
├── config.py                   # Configuration constants (26 lines)
├── pipeline.py                 # GStreamer pipeline + lifecycle (482 lines)
├── main.py                     # Flask HTTP API (103 lines)
├── Dockerfile                  # Builds ds-single:latest
├── libnvdsinfer_custom_impl_Yolo.so  # YOLOv8 custom parser library
├── config_tracker_NvDCF_perf.yml    # Tracker config (unused currently)
│
├── config/
│   └── bowling_yolo12n_batch.txt   # YOLO12n bowling @ 1280, batch-size=36
│
├── models/
│   ├── *.onnx                  # ONNX models
│   ├── *.txt                   # Label files
│   └── *.engine                # TensorRT engine cache (auto-generated, gitignored)
│
├── training/                   # Training artifacts (gitignored)
│   ├── scripts/
│   │   └── export_deepstream.sh  # DeepStream-Yolo ONNX export wrapper
│   ├── DeepStream-Yolo/        # Custom ONNX export scripts (cloned repo)
│   └── runs/                   # Training runs
│
└── infra/                      # Helper scripts (optional, may not exist)
    ├── restart_stream.sh       # curl wrapper for /stream/restart
    └── scripts/
        └── ...                 # Utility scripts
```

## Key Files

### Source Code (Python)
- **config.py** (26 lines): All configuration constants
  - Pipeline configuration (GPU_ID, MAX_NUM_SOURCES, dimensions)
  - Network configuration (RTSP_SERVER_PORT=9600, HTTP_API_PORT=5555)
  - Encoder settings (bitrate, IDR interval)

- **pipeline.py** (482 lines): GStreamer pipeline and lifecycle management
  - Pipeline creation/destruction functions
  - Source management (add/remove/restart)
  - Output bin creation
  - RTSP server and factory management
  - All GStreamer callbacks

- **main.py** (103 lines): Flask HTTP API only
  - POST /stream/restart - Start/restart stream by ID
  - GET /stream/status - Get active streams and pipeline status
  - Lifecycle coordination (0→1 creates pipeline)

### Build and Run Scripts
- **build.sh**: Builds ds-single:latest Docker image
  - Stops old ds-single container
  - Builds new image with updated Python code

- **run.sh**: Starts ds-single container
  - Stops old ds-single container
  - Starts new container with volume mounts
  - Shows startup logs

### Configuration Files
- **config/bowling_yolo12n_batch.txt**: YOLOv8 inference config
  - Model path: `/models/bowling_yolo12n.onnx`
  - Engine path: `/models/bowling_yolo12n_1280_batch36.engine`
  - Batch size: 36
  - Network mode: 2 (FP16)

### Docker
- **Dockerfile**: Multi-stage build
  - Base: nvcr.io/nvidia/deepstream:8.0-triton-multiarch
  - Installs: python3-flask, python3-gi, DeepStream Python bindings
  - Copies: config.py, pipeline.py, main.py, custom parser
  - CMD: Runs main.py with 3 source URIs (in_s0, in_s1, in_s2)

## Architecture Summary

### Single Container System
- **ds-single container**: One container, up to 36 dynamic streams
- **HTTP API**: Flask on port 5555 for stream control
- **RTSP server**: GstRtspServer on port 9600, paths /x0-/x35
- **Pipeline**: Created lazily on first /stream/restart call

### Data Flow
```
HTTP POST /stream/restart {"id": N}
         ↓
Flask (main.py) checks if pipeline exists
         ↓
If first stream (0→1): pipeline.create_pipeline(uris)
         ↓
GStreamer pipeline starts (pipeline.py)
         ↓
uridecodebin pulls RTSP → nvstreammux → nvinfer (YOLOv8) → nvstreamdemux
         ↓
Per-stream output chain → encoder → udpsink (port 5001+N)
         ↓
GstRtspServer exposes rtsp://localhost:9600/xN
```

### Code Flow
```
Container starts → main.py runs
         ↓
Flask starts listening on :5555
         ↓
User: POST /stream/restart {"id": 0}
         ↓
main.py: if pipeline.pipeline is None → pipeline.create_pipeline()
         ↓
pipeline.py: Creates GStreamer pipeline with first source
         ↓
User: POST /stream/restart {"id": 1}
         ↓
main.py: pipeline.restart_source(1) (pipeline already exists)
         ↓
pipeline.py: Adds second source to existing pipeline
```

## Typical Workflows

### Start Container
```bash
./build.sh         # Only if Python code changed
./run.sh          # Start ds-single container
```

### Start First Stream (Creates Pipeline)
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"id": 0}' \
  http://localhost:5555/stream/restart

# Or use helper script (if it exists)
./infra/restart_stream.sh 0
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
docker logs ds-single --follow
```

### Stop Container
```bash
docker stop ds-single
# Or restart with:
./run.sh
```

### Change Model
```bash
# 1. Edit config.py to point to new config file
vim config.py
# Update: PGIE_CONFIG_FILE = "/config/new_model_config.txt"

# 2. Rebuild and restart
./build.sh
./run.sh

# DeepStream will auto-build new engine on first run
```

### Manage TensorRT Engine Cache
```bash
# Engines are auto-managed in /models/ (bind-mounted)
# First run: builds engine (3-10 min)
# Subsequent runs: loads cached engine (instant)

# To force rebuild:
rm models/*.engine
./run.sh
```

## What NOT to Do

1. **Don't edit files inside container**: Edit on host, rebuild with `./build.sh`
2. **Don't manually create RTSP factories**: They're created dynamically by pipeline.py
3. **Don't skip rebuilding after code changes**: Docker uses cached image
4. **Don't forget volume mounts**: /config and /models must be mounted (see run.sh)

## File Locations Summary

**Source code** (edited on host):
- `/root/d_final/config.py`
- `/root/d_final/pipeline.py`
- `/root/d_final/main.py`

**Inside container** (after build):
- `/app/config.py`
- `/app/pipeline.py`
- `/app/main.py`

**Bind-mounts** (shared host↔container):
- `/root/d_final/config/` → `/config/` (inference configs)
- `/root/d_final/models/` → `/models/` (ONNX + TensorRT engines)

**Ports**:
- 5555: HTTP API (Flask)
- 9600: RTSP server (GstRtspServer)
- 5001-5036: UDP sinks (internal, used by RTSP server)

## Training Directory

```
training/
├── scripts/
│   └── export_deepstream.sh              # ONNX export wrapper
├── DeepStream-Yolo/                      # Export scripts repo
└── runs/                                 # Training outputs
    └── train/
        └── weights/
            ├── best.pt                   # PyTorch checkpoint
            └── best.onnx                 # Exported ONNX
```

**Key Points:**
- Training runs are gitignored (except metadata)
- Production models copied to `/root/d_final/models/` and committed
- Use Ultralytics export with `dynamic=False` for DeepStream compatibility
