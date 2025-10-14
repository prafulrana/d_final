# Project Structure

```
d_final/
├── app.py                    # Main DeepStream application (318 lines)
├── Dockerfile                # DeepStream 8.0 Python environment
├── up.sh                     # Build and run script (named container)
├── models/                   # Persistent TensorRT engine cache (volume mounted)
├── plan.md                   # Implementation history and current status
└── AGENTS.md                 # Guidelines for AI agents
```

## Key Files

### app.py
Main Python DeepStream application implementing:
- **Pipeline**: nvurisrcbin → nvstreammux → nvinfer → nvvideoconvert → nvdsosd → nvvideoconvert → capsfilter → nvv4l2h264enc → **queue** → h264parse → rtspclientsink
- **Input**: RTSP stream with automatic reconnection via nvurisrcbin
- **Output**: RTSP stream with detection boxes overlay
- **Reconnection**: Built-in nvurisrcbin with TCP-only protocol (no manual logic)

**Key Sections**:
- Lines 16-37: `bus_call()` - handles stream-eos messages (informational only)
- Lines 39-62: `cb_newpad()` - sets ghost pad target when decoder pad appears
- Lines 64-67: `decodebin_child_added()` - tracks decodebin children
- Lines 69-108: `create_source_bin()` - wraps nvurisrcbin in Bin with ghost pad
- Lines 110-332: `main()` - creates pipeline, links elements, starts event loop
- Lines 315-332: `parse_args()` and entry point

**Critical Optimizations** (added for smooth output):
- Lines 201-205: Encoder properties (preset-id=0, profile=2)
- Lines 207-212: Queue element (essential for smooth frame delivery)
- Line 230: rtspclientsink latency=200ms (smooth RTP timing)

**nvurisrcbin Configuration** (lines 86-91):
```python
uri_decode_bin.set_property("rtsp-reconnect-interval", 10)
uri_decode_bin.set_property("init-rtsp-reconnect-interval", 5)
uri_decode_bin.set_property("rtsp-reconnect-attempts", -1)  # Infinite
uri_decode_bin.set_property("select-rtp-protocol", 4)  # TCP-only
```

### Dockerfile
Based on `nvcr.io/nvidia/deepstream:8.0-triton-multiarch`:
- Installs DeepStream Python bindings via official script
- Copies ONNX model to /models/ inside image
- Configures pgie_config.txt to use /models/ paths
- Enables GStreamer debug logging (GST_DEBUG=2)
- Runs app.py with hardcoded RTSP URLs (can override via CMD)

### up.sh
Multi-pipeline orchestration script:
- Builds Docker image: `ds_python:latest`
- Cleans up all old containers using the image
- Runs **6 concurrent inference pipelines** on single input stream:
  - **drishti-s0**: ResNet Traffic → s0
  - **drishti-s1**: PeopleNet INT8 → s1
  - **drishti-s2**: ResNet Detector → s2
  - **drishti-s3**: City Segmentation → s3
  - **drishti-s4**: People Segmentation → s4
  - **drishti-s5**: Default Test1 → s5
- Each container:
  - GPU access: `--gpus all` (shared GPU)
  - Network: `--network host`
  - Volumes: `-v models:/models` (TensorRT cache), `-v config:/config` (inference configs)
  - Reads from: `rtsp://relay:8554/in_s0`
  - Publishes to: `rtsp://relay:8554/s{0-5}`

### models/
Persistent directory (volume mounted from host):
- **Purpose**: Cache TensorRT engine to avoid 20+ second rebuilds
- **Contents**: `resnet18_trafficcamnet_pruned.onnx` and `.engine` files
- **Why**: TensorRT engine is GPU-specific and takes ~38 seconds to build
- **Result**: Startup reduced from ~38s to ~5s

## Architecture Details

### Ghost Pad Pattern
nvurisrcbin dynamically creates pads after connecting to RTSP source. The ghost pad pattern abstracts this:

1. Create Bin wrapper around nvurisrcbin
2. Add ghost pad with no target initially
3. Connect `pad-added` signal
4. When decoder pad appears with NVMM caps, set ghost pad target
5. Link Bin's ghost pad to nvstreammux sink pad

This follows NVIDIA's official pattern from `deepstream-test3`.

### TCP-Only Protocol
Critical for fast reconnection:
- **Property**: `select-rtp-protocol=4` (RTP_PROTOCOL_TCP)
- **Why**: Default value 7 (UDP+TCP) tries UDP first, waits 5 seconds for timeout
- **Impact**: Eliminates 5-second UDP timeout on every reconnection attempt
- **Result**: Fast recovery from both clean disconnects and abrupt closes

### Pipeline Flow
```
RTSP Input (in_s0)
  ↓
nvurisrcbin (automatic reconnection)
  ↓
Bin with ghost pad
  ↓
nvstreammux (batch-size=1, 1280x720)
  ↓
nvinfer (TrafficCamNet ONNX → TensorRT)
  ↓
nvvideoconvert (pre-OSD)
  ↓
nvdsosd (draw detection boxes)
  ↓
nvvideoconvert (post-OSD, format conversion)
  ↓
capsfilter (I420 format)
  ↓
nvv4l2h264enc (H.264 encoding, preset-id=0/P1, profile=2/Main, bitrate=3Mbps)
  ↓
queue (frame buffering for smooth delivery) ← CRITICAL for smooth output
  ↓
h264parse (config-interval=-1)
  ↓
rtspclientsink (TCP-only, latency=200ms) ← latency CRITICAL for RTP timing
  ↓
RTSP Output (s0)
```

## Command Line Arguments

```bash
python3 app.py -i <rtsp_input> -o <rtsp_output> [-c <pgie_config>]
```

**Example** (from Dockerfile CMD):
```bash
python3 -u /app/app.py \
  -i rtsp://34.100.230.7:8554/in_s0 \
  -o rtsp://34.100.230.7:8554/s0 \
  -c /app/pgie_config.txt
```

## Typical Usage

```bash
# Build and start
./up.sh

# View logs
docker logs -f drishti

# Stop
docker rm -f drishti

# Publish test stream to in_s0 (Larix, gst-launch, ffmpeg)
# View output at rtsp://server:8554/s0 (with detection boxes)
```

## Performance Baseline

**Proven Capability**: This hardware handles 64 concurrent 1080p RTSP streams with full TensorRT inference (TrafficCamNet).

**Therefore**: Any performance issues with a single 720p stream are NOT due to:
- Resolution being too high
- Inference being too heavy
- GPU compute limits
- Encoding bandwidth

**Actual causes are always**:
- Configuration (buffer timeouts, queue sizes)
- Timing (startup delays, engine rebuild)
- Network (latency, packet loss)
- Memory management (not using NVMM)

## Related Projects

This is a simplified single-stream version. For multi-stream batch processing, see:
- `/root/d/batch_streaming_v1/` - C implementation with nvmultiurisrcbin
- Handles 64+ concurrent streams with dynamic add/remove
- Uses watchdog monitoring MediaMTX API
