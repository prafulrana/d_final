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
- **Pipeline**: nvurisrcbin → nvstreammux → nvinfer → nvvideoconvert → nvdsosd → nvvideoconvert → capsfilter → nvv4l2h264enc → h264parse → rtspclientsink
- **Input**: RTSP stream with automatic reconnection via nvurisrcbin
- **Output**: RTSP stream with detection boxes overlay
- **Reconnection**: Built-in nvurisrcbin with TCP-only protocol (no manual logic)

**Key Sections**:
- Lines 16-37: `bus_call()` - handles stream-eos messages (informational only)
- Lines 39-62: `cb_newpad()` - sets ghost pad target when decoder pad appears
- Lines 64-67: `decodebin_child_added()` - tracks decodebin children
- Lines 69-108: `create_source_bin()` - wraps nvurisrcbin in Bin with ghost pad
- Lines 110-298: `main()` - creates pipeline, links elements, starts event loop
- Lines 300-317: `parse_args()` and entry point

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
Orchestration script:
- Builds Docker image: `ds_python:latest`
- Cleans up old containers by name: `drishti`
- Kills any orphaned containers using the image
- Runs container in detached mode with:
  - Named: `drishti`
  - GPU access: `--gpus all`
  - Network: `--network host`
  - Volume: `-v $PWD/models:/models` (engine persistence)

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
nvv4l2h264enc (H.264 encoding, bitrate=3Mbps)
  ↓
h264parse (config-interval=-1)
  ↓
rtspclientsink (TCP-only)
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

## Related Projects

This is a simplified single-stream version. For multi-stream batch processing, see:
- `/root/d/batch_streaming_v1/` - C implementation with nvmultiurisrcbin
- Handles 64+ concurrent streams with dynamic add/remove
- Uses watchdog monitoring MediaMTX API
