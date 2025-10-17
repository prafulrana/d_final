# Project Structure

```
d_final/
├── main.cpp                              # Pure C++ GStreamer YOLO detection application
├── libnvdsinfer_custom_impl_Yolo.so      # Custom YOLO parser library
├── build_app.sh                          # Builds C++ application
├── Dockerfile                            # DeepStream 8.0 C++ build environment
├── build.sh                              # Build Docker image
├── up.sh                                 # Start 3 YOLO detection pipelines (s0, s1, s2)
├── config/
│   ├── pgie_yolov8_coco.txt             # YOLOv8 COCO detection config (2048x2048)
│   └── tracker_smooth.yml               # NvDCF tracker config (smooth bboxes)
├── scripts/
│   ├── export_yolov8.sh                 # Export YOLOv8 to DeepStream ONNX format
│   ├── download_model.sh                # Download models from YOLOv8/NGC/Roboflow
│   └── cache_engine.sh                  # Manage TensorRT engine cache
├── models/                               # Model files (Git LFS)
│   ├── yolov8n_2048.onnx                # YOLOv8 nano 2048x2048 (13MB, DeepStream format)
│   ├── yolov8n.pt                       # PyTorch weights (6.3MB)
│   ├── coco_labels.txt                  # COCO class labels (80 classes)
│   └── *.engine                         # TensorRT cache (not in git)
├── DeepStream-Yolo/                      # Community YOLO support (cloned)
│   ├── nvdsinfer_custom_impl_Yolo/      # Custom parser source
│   └── utils/export_yoloV8.py           # DeepStream-specific ONNX export
├── relay/                                # MediaMTX relay infrastructure (GCP)
│   ├── main.tf                          # Terraform config
│   ├── variables.tf                     # Variables
│   ├── scripts/startup.sh               # VM startup script
│   └── README.md                        # Deployment guide
├── AGENTS.md                             # AI agent guidelines
├── STANDARDS.md                          # Development standards
└── STRUCTURE.md                          # This file
```

## Architecture Overview

### DeepStream Detection Pipelines

**Three concurrent YOLO detection pipelines** (s0, s1, s2):

| Pipeline | Model | Resolution | Input | Output | Purpose |
|----------|-------|-----------|-------|--------|---------|
| **s0** | YOLOv8n COCO | 2048x2048 | `in_s0` | `s0` | Object detection with tracking |
| **s1** | YOLOv8n COCO | 2048x2048 | `in_s1` | `s1` | Object detection with tracking |
| **s2** | YOLOv8n COCO | 2048x2048 | `in_s2` | `s2` | Object detection with tracking |

- All 3 streams share the same TensorRT engine for fast startup
- NvDCF multi-object tracker for stable, smooth bounding boxes
- 4Mbps H.264 encoding with preset-level 3 for quality

### Pipeline Flow

```
nvurisrcbin (RTSP input, TCP reconnect)
  ↓
nvstreammux (batch=1, 1280x720, 40ms timeout)  ← Low latency batching
  ↓
nvinfer (YOLOv8 2048x2048, interval=0)         ← Inference every frame
  ↓
nvtracker (NvDCF with smooth Kalman filter)    ← Stable tracking
  ↓
nvvidconv (NV12 → RGBA)
  ↓
nvosd (draw boxes, tracking IDs, CPU mode)
  ↓
capsfilter (RGBA)
  ↓
nvvidconv_postosd (RGBA → I420)
  ↓
capsfilter (I420 format)
  ↓
nvv4l2h264enc (4Mbps CBR, preset 3, I-frame 30)
  ↓
queue (4 buffer)
  ↓
h264parse
  ↓
rtspclientsink (RTSP output, TCP, 100ms latency)
```

## Key Files

### main.cpp

Pure C++ GStreamer application with NvDCF tracking:

**Key optimizations:**
- `batched-push-timeout: 40000` (40ms) - Smooth frame delivery
- `interval: 0` - Inference every frame
- Custom tracker config with higher process noise for smooth boxes
- CBR encoding with slow preset for clean motion

**Tracker configuration:**
```cpp
g_object_set(G_OBJECT(nvtracker), "ll-config-file", "/config/tracker_smooth.yml", NULL);
g_object_set(G_OBJECT(nvtracker), "tracker-width", 640, NULL);
g_object_set(G_OBJECT(nvtracker), "tracker-height", 384, NULL);
```

**Encoder configuration:**
```cpp
g_object_set(G_OBJECT(encoder), "bitrate", 4000000, NULL);     // 4Mbps
g_object_set(G_OBJECT(encoder), "iframeinterval", 30, NULL);   // 1 second
g_object_set(G_OBJECT(encoder), "control-rate", 1, NULL);      // CBR
g_object_set(G_OBJECT(encoder), "preset-level", 3, NULL);      // Slow = quality
```

### config/pgie_yolov8_coco.txt

YOLOv8 inference configuration:

```ini
onnx-file=/models/yolov8n_2048.onnx
model-engine-file=/models/yolov8n_2048_b1_gpu0_fp16.engine
batch-size=1
network-mode=2                    # FP16
interval=0                        # Inference every frame
network-type=0                    # Detector
maintain-aspect-ratio=0
pre-cluster-threshold=0.15        # Lower threshold for more detections
nms-iou-threshold=0.45
```

### config/tracker_smooth.yml

NvDCF tracker with smooth bounding boxes:

**Key parameters:**
- `processNoiseVar4Loc: 3.0` - Higher = smoother position (default: 1.5)
- `processNoiseVar4Size: 3.0` - Higher = smoother size (default: 1.3)
- `filterLr: 0.02` - Lower = smoother visual tracking (default: 0.075)
- `maxShadowTrackingAge: 60` - Keep tracks alive 60 frames without detection

### scripts/export_yolov8.sh

Automated YOLOv8 export to DeepStream ONNX format:

```bash
# Usage: ./scripts/export_yolov8.sh <model> <resolution>
./scripts/export_yolov8.sh yolov8n 2048
./scripts/export_yolov8.sh yolov8s 1280
./scripts/export_yolov8.sh yolov8m 1024
```

**Process:**
1. Downloads YOLOv8 weights if not exists
2. Exports to ONNX with DeepStream-specific format
3. Outputs: `models/yolov8n_<resolution>.onnx`

### scripts/download_model.sh

Download models from various sources:

```bash
# YOLOv8 models
./scripts/download_model.sh yolov8 yolov8n
./scripts/download_model.sh yolov8 yolov8s

# NGC models (requires ngc-cli)
./scripts/download_model.sh ngc nvidia/tao/peoplesegnet:deployable_v2.0.2
```

### up.sh

Pipeline orchestration - starts 3 concurrent streams:

```bash
# All 3 streams use same model and engine
docker run -d --name drishti-s0 --gpus all --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/libnvdsinfer_custom_impl_Yolo.so":/app/libnvdsinfer_custom_impl_Yolo.so \
  ds_python:latest \
  /app/deepstream_app \
  rtsp://$RELAY_IP:8554/in_s0 \
  rtsp://$RELAY_IP:8554/s0 \
  /config/pgie_yolov8_coco.txt
```

**Volume mounts:**
- `models/`: TensorRT engine cache shared across all 3 streams
- `config/`: Inference and tracker configs
- `libnvdsinfer_custom_impl_Yolo.so`: Custom YOLO parser

### models/

**YOLO Detection**:
- `yolov8n_2048.onnx` (13MB) - YOLOv8 nano 2048x2048 resolution
  - **Exported with**: `DeepStream-Yolo/utils/export_yoloV8.py`
  - **NOT** standard Ultralytics export - uses custom DeepStreamOutput layer
- `yolov8n.pt` (6.3MB) - PyTorch weights
- `coco_labels.txt` - 80 COCO class labels

**TensorRT cache**:
- `yolov8n_2048_b1_gpu0_fp16.engine` (~12MB) - GPU-specific engine
- Built on first run (~5-10 min for 2048x2048)
- Shared by all 3 streams for instant startup

### scripts/cache_engine.sh

TensorRT engine management tool:

```bash
# List all engines (containers and host)
./scripts/cache_engine.sh list

# Copy engine from container to host (after first build)
./scripts/cache_engine.sh copy drishti-s0

# Verify engine exists for current config
./scripts/cache_engine.sh verify config/pgie_yolov8_coco.txt

# Clean all cached engines
./scripts/cache_engine.sh clean
```

**Typical workflow:**
1. First run: `./up.sh` (builds engine 5-10 min)
2. Monitor: `docker logs -f drishti-s0` (watch for "Running main loop...")
3. Cache: `./scripts/cache_engine.sh copy drishti-s0`
4. Future runs: Instant startup using cached engine

**Why cache engines:**
- TensorRT engines are GPU-specific (must rebuild on different GPU)
- Building takes 5-10 minutes for 2048x2048 models
- All 3 streams share the same engine
- Volume-mounted `/models` persists cache across container restarts

## Development Workflow

### Adding New Model Resolution

```bash
# 1. Export ONNX at desired resolution
./scripts/export_yolov8.sh yolov8n 1280

# 2. Update config
vim config/pgie_yolov8_coco.txt
# Change: onnx-file=/models/yolov8n_1280.onnx
# Change: model-engine-file=/models/yolov8n_1280_b1_gpu0_fp16.engine

# 3. Restart pipelines (will build engine on first run)
./up.sh
```

### Upgrading to Larger Model

```bash
# 1. Download and export YOLOv8s
./scripts/download_model.sh yolov8 yolov8s
./scripts/export_yolov8.sh yolov8s 2048

# 2. Update config
vim config/pgie_yolov8_coco.txt
# Change: onnx-file=/models/yolov8s_2048.onnx
# Change: model-engine-file=/models/yolov8s_2048_b1_gpu0_fp16.engine

# 3. Restart
./up.sh
```

### C++ Changes (requires rebuild)

```bash
# Edit source
vim main.cpp

# Rebuild and restart
./build.sh && ./up.sh
```

### Config Changes (no rebuild)

```bash
# Edit config
vim config/pgie_yolov8_coco.txt
vim config/tracker_smooth.yml

# Just restart
./up.sh
```

### Debugging

```bash
# Check logs
docker logs -f drishti-s0

# Check container status
docker ps -a | grep drishti

# View output streams
http://34.14.140.30:8889/s0/
http://34.14.140.30:8889/s1/
http://34.14.140.30:8889/s2/
```

## Performance Notes

**Current setup:**
- 3 concurrent streams @ 2048x2048 YOLOv8n
- Hardware: NVIDIA 5070Ti/5080
- Real-time inference (30+ FPS per stream)
- Shared TensorRT engine for fast startup

**Optimizations:**
- Low-latency batching (40ms timeout)
- Inference every frame (interval=0)
- Smooth tracking (higher process noise variance)
- Quality encoding (preset-level 3, 4Mbps CBR)

## Common Issues

### Bounding Box Jitter

**Symptom**: Boxes resize/move slightly every frame

**Fix**: Increase tracker smoothing in `config/tracker_smooth.yml`:
```yaml
StateEstimator:
  processNoiseVar4Loc: 3.0    # Higher = smoother position
  processNoiseVar4Size: 3.0   # Higher = smoother size
```

### Motion Corruption/Pixelation

**Symptom**: Blocky artifacts during fast motion

**Fix**: Increase bitrate or use slower preset:
```cpp
g_object_set(G_OBJECT(encoder), "bitrate", 8000000, NULL);     // 8Mbps
g_object_set(G_OBJECT(encoder), "preset-level", 4, NULL);      // Very slow
```

### Jerky/Non-smooth Frames

**Symptom**: Frames appear to batch/stutter

**Fix**: Lower nvstreammux timeout in `main.cpp`:
```cpp
g_object_set(G_OBJECT(nvstreammux), "batched-push-timeout", 40000, NULL);  // 40ms
```

### Object Flickering (On/Off Detection)

**Fix 1**: Lower detection threshold in `config/pgie_yolov8_coco.txt`:
```ini
pre-cluster-threshold=0.10    # Lower for more detections
```

**Fix 2**: Increase shadow tracking in `config/tracker_smooth.yml`:
```yaml
TargetManagement:
  maxShadowTrackingAge: 90    # Keep tracks alive longer
```

## Related Documentation

- **AGENTS.md**: AI agent guidelines, pipeline patterns, troubleshooting
- **STANDARDS.md**: Development standards, testing workflow
- **relay/README.md**: MediaMTX relay deployment guide
