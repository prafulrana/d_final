# Project Structure

```
d_final/
├── app.py                      # Main DeepStream application with multi-probe support
├── probe_default.py            # Default probe (no custom processing)
├── probe_yoloworld.py          # YOLOWorld custom tensor parsing
├── probe_segmentation.py       # Custom CUDA segmentation overlay
├── segmentation_overlay.cu     # CUDA kernel for green overlay
├── Dockerfile                  # DeepStream 8.0 Python + CUDA build environment
├── build.sh                    # Build Docker image
├── build_cuda.sh               # Compile CUDA kernel
├── up.sh                       # Start 3 inference pipelines (s0, s1, s2)
├── down.sh                     # Stop all pipelines
├── config/
│   ├── README.md               # Config documentation
│   ├── pgie_resnet_traffic.txt # TrafficCamNet config (s0)
│   ├── pgie_yoloworld.txt      # YOLOWorld config (s1)
│   └── pgie_peoplesegnet.txt   # PeopleSemSegNet config (s2)
├── models/                     # Model files (Git LFS)
│   ├── resnet18_trafficcamnet_pruned.onnx
│   ├── yoloworld_custom.onnx
│   ├── peoplesemsegnet_shuffleseg.onnx
│   ├── labels.txt
│   └── *.engine               # TensorRT cache (not in git)
├── relay/                      # MediaMTX relay infrastructure (GCP)
│   ├── main.tf                # Terraform config
│   ├── variables.tf           # Variables
│   ├── scripts/startup.sh     # VM startup script
│   └── README.md              # Deployment guide
├── AGENTS.md                   # AI agent guidelines
├── STANDARDS.md                # Development standards
└── STRUCTURE.md                # This file
```

## Architecture Overview

### Multi-Probe System

**3 concurrent inference pipelines**, each with a different inference model and probe:

| Pipeline | Model | Probe | Input | Output | Purpose |
|----------|-------|-------|-------|--------|---------|
| **s0** | TrafficCamNet | `probe_default` | `in_s0` | `s0` | Standard object detection (Vehicle, Person, RoadSign, TwoWheeler) |
| **s1** | YOLOWorld | `probe_yoloworld` | `in_s1` | `s1` | Custom tensor parsing with COCO classes |
| **s2** | PeopleSemSegNet | `probe_segmentation` | `in_s2` | `s2` | Custom CUDA green overlay for people segmentation |

### Pipeline Order (CRITICAL)

```
nvurisrcbin (RTSP input)
  ↓
Bin with ghost pad
  ↓
nvstreammux (batch=1, 1280x720)
  ↓
nvinfer (ONNX → TensorRT)
  ↓
nvvidconv (NV12 → RGBA)          ← MUST come before nvosd
  ↓
nvosd (draw boxes, finalize segmentation metadata)
  ↓
rgba_caps (probe attachment point varies)
  ↓
nvvidconv_postosd (RGBA → I420)
  ↓
capsfilter (I420 format)
  ↓
nvv4l2h264enc (H.264 encoding)
  ↓
queue (frame buffering)          ← CRITICAL for smooth output
  ↓
h264parse
  ↓
rtspclientsink (RTSP output)
```

**Why this order matters:**
- **nvvidconv BEFORE nvosd**: nvosd requires RGBA format to draw (CPU mode)
- **queue after encoder**: Prevents choppy video by buffering frames

### Probe Attachment Patterns

Different probe types attach at different points:

**probe_yoloworld** (custom tensor parsing):
- Attaches to: `nvvidconv.get_static_pad("src")` (BEFORE nvosd)
- Why: Must add `obj_meta` before nvosd draws boxes
- Process: Parse tensor → create obj_meta → nvosd draws

**probe_segmentation** (custom CUDA overlay):
- Attaches to: `rgba_caps.get_static_pad("sink")` (AFTER nvosd)
- Why: Needs finalized segmentation metadata from nvosd
- Process: Get seg_meta → apply CUDA overlay → bypass nvosd drawing

**probe_default** (no-op):
- Attaches to: `rgba_caps.get_static_pad("sink")` (AFTER nvosd)
- Does nothing, nvosd handles all drawing

## Key Files

### app.py (328 lines)

Main DeepStream application with multi-probe architecture:

**Key Functions:**
- `bus_call()` (lines 16-37): Handles pipeline messages
- `cb_newpad()` (lines 39-62): Sets ghost pad target when decoder pad appears
- `create_source_bin()` (lines 69-108): Wraps nvurisrcbin with ghost pad
- `main()` (lines 110-340): Creates pipeline, conditionally attaches probe

**Probe Loading:**
```python
probe_module = __import__(args.probe)  # Dynamic import

# Conditional attachment based on probe type
if args.probe == "probe_yoloworld":
    nvvidconv_srcpad = nvvidconv.get_static_pad("src")
    nvvidconv_srcpad.add_probe(Gst.PadProbeType.BUFFER, probe_module.osd_sink_pad_buffer_probe, 0)
else:
    rgba_sinkpad = rgba_caps.get_static_pad("sink")
    rgba_sinkpad.add_probe(Gst.PadProbeType.BUFFER, probe_module.osd_sink_pad_buffer_probe, 0)
```

**nvurisrcbin Configuration:**
```python
uri_decode_bin.set_property("rtsp-reconnect-interval", 10)
uri_decode_bin.set_property("init-rtsp-reconnect-interval", 5)
uri_decode_bin.set_property("rtsp-reconnect-attempts", -1)  # Infinite
uri_decode_bin.set_property("select-rtp-protocol", 4)       # TCP-only
```

**Critical Optimizations:**
```python
encoder.set_property("preset-id", 0)       # P1 (highest performance)
encoder.set_property("profile", 2)         # Main profile
rtsp_sink.set_property("latency", 200)     # 200ms buffer for smooth RTP
```

### Probe Modules

**probe_default.py** (7 lines):
- No-op probe, returns immediately
- nvosd handles all drawing

**probe_yoloworld.py** (137 lines):
- Parses raw tensor output [N, 6]: `[x1, y1, x2, y2, score, class]`
- Uses `maintain-aspect-ratio=0` for simple resize (no letterboxing)
- Direct coordinate scaling: `x_frame = x_net * (frame_width / 640)`
- Creates `obj_meta` with bounding boxes and COCO class labels
- nvosd draws the boxes

**probe_segmentation.py** (93 lines):
- Accesses segmentation metadata via `pyds.NvDsInferSegmentationMeta`
- Converts int32 mask to uint8: `mask_array.astype(np.uint8)`
- Allocates GPU memory and copies mask
- Calls CUDA kernel to apply green overlay (50% alpha)
- Bypasses nvosd drawing (custom visualization)

### segmentation_overlay.cu (68 lines)

CUDA kernel for people segmentation overlay:
- Input: RGBA frame (NVMM), uint8 mask (class 0=background, 1=person)
- Output: Green translucent overlay on person pixels
- Thread grid: 16x16 blocks covering full frame resolution
- Alpha blending: `new = original * (1-alpha) + green * alpha`

### up.sh

Multi-pipeline orchestration script:

1. Stops all existing containers
2. Builds `ds_python:latest` Docker image
3. Starts 3 concurrent pipelines:

```bash
# s0: TrafficCamNet with standard detection
docker run -d --name drishti-s0 --gpus all --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/app.py":/app/app.py \
  -v "$(pwd)/probe_default.py":/app/probe_default.py \
  ds_python:latest python3 -u /app/app.py \
    -i rtsp://RELAY_IP:8554/in_s0 \
    -o rtsp://RELAY_IP:8554/s0 \
    -c /config/pgie_resnet_traffic.txt

# s1: YOLOWorld with custom tensor parsing
docker run -d --name drishti-s1 --gpus all --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/app.py":/app/app.py \
  -v "$(pwd)/probe_yoloworld.py":/app/probe_yoloworld.py \
  ds_python:latest python3 -u /app/app.py \
    -i rtsp://RELAY_IP:8554/in_s1 \
    -o rtsp://RELAY_IP:8554/s1 \
    -c /config/pgie_yoloworld.txt \
    --probe probe_yoloworld

# s2: PeopleSemSegNet with custom CUDA overlay
docker run -d --name drishti-s2 --gpus all --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/app.py":/app/app.py \
  -v "$(pwd)/probe_segmentation.py":/app/probe_segmentation.py \
  ds_python:latest python3 -u /app/app.py \
    -i rtsp://RELAY_IP:8554/in_s2 \
    -o rtsp://RELAY_IP:8554/s2 \
    -c /config/pgie_peoplesegnet.txt \
    --probe probe_segmentation
```

**Volume Mounts:**
- `models/`: TensorRT engine cache (persistent across runs)
- `config/`: Inference configurations (can modify without rebuild)
- `app.py`, `probe_*.py`: Python files (fast iteration - restart only)

### Dockerfile

Multi-stage build:
1. **Base**: DeepStream 8.0 Triton Multiarch
2. **CUDA stage**: Compiles `segmentation_overlay.cu` → `.so`
3. **Final stage**: Installs Python bindings, copies CUDA library

**Key features:**
- DeepStream Python bindings via official install script
- PyCUDA for GPU memory management
- CUDA kernel compiled with `nvcc`
- Python files copied last (frequent changes)

### models/ (Git LFS)

**Active models** (737MB total):
- `resnet18_trafficcamnet_pruned.onnx` (5.2MB) - 4 classes: Vehicle, Person, RoadSign, TwoWheeler
- `yoloworld_custom.onnx` (277MB) - 80 COCO classes with custom architecture
- `peoplesemsegnet_shuffleseg.onnx` (3.8MB) - Binary segmentation (background, person)
- `labels.txt` - Label file for PeopleSemSegNet

**TensorRT cache** (generated, not in git):
- `*.engine` files - GPU-specific, ~5-10s to generate on first run
- Stored in `/models/` via volume mount for persistence

## Development Workflow

### Fast Iteration (Python changes)

Python files are **volume mounted**, no rebuild needed:

```bash
# Edit probe_segmentation.py or app.py
vim probe_segmentation.py

# Just restart container (volume mount takes effect)
docker restart drishti-s2

# Changes applied immediately!
```

**Important:** Python bytecode caching can cause stale imports:
```bash
docker exec drishti-s2 find /app -name "*.pyc" -delete
docker restart drishti-s2
```

### Full Rebuild (CUDA/Dockerfile changes)

```bash
# CUDA or Dockerfile changes require rebuild
./build.sh      # Rebuild image
./up.sh         # Restart all pipelines
```

### Testing Changes

```bash
./down.sh       # Stop all pipelines
# Make changes
./up.sh         # Rebuild and restart all

# Check logs
docker logs drishti-s0
docker logs drishti-s1
docker logs drishti-s2

# View outputs (WebRTC)
http://RELAY_IP:8889/s0/
http://RELAY_IP:8889/s1/
http://RELAY_IP:8889/s2/
```

## Performance Notes

**Proven capability**: This hardware handles 64+ concurrent 1080p streams with TensorRT inference.

**Current setup**: 3 concurrent 720p streams is trivial - any issues are configuration, not performance.

## Related Documentation

- **AGENTS.md**: AI agent guidelines, pipeline patterns, common pitfalls
- **STANDARDS.md**: Development standards, testing workflow
- **relay/README.md**: MediaMTX relay deployment guide
