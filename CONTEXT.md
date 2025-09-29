# DeepStream AI Pipeline Project - Developer Context

## 🎯 PRIMARY GOAL: 10-SECOND VIDEO WITH AI DETECTION

**OBJECTIVE:** Ensure `npm test` passes completely and produces a 10-second video output showing AI detection bounding boxes.

**SUCCESS CRITERIA:**
- Golden flow completes without hanging
- Video file generated in `/outputs/` directory
- AI bounding boxes clearly visible on traffic/vehicles
- File size > 1MB (indicates real content, not empty)

**ANTI-RETARD FOR VIDEO OUTPUT:**
- ❌ Don't focus on dual streams if single stream works
- ❌ Don't debug RTSP if file inputs work better for proof
- ❌ Don't get stuck on cam2 - prove concept with cam1 first
- ✅ Priority: Working video with visible AI detection
- ✅ Modify config to single stream if needed for success
- ✅ Use file:// URLs if RTSP continues failing

## 🚨 CRITICAL ANTI-RETARD RULES - READ FIRST 🚨

### ❌ DOCKER CLI FORBIDDEN ❌
- **NEVER** use `docker run`, `docker rm`, `docker ps` commands
- **NEVER** use `docker kill`, `docker stop` via command line
- **ONLY** use Docker Socket API with axios for ALL container operations
- **ALL** cleanup must be API-driven through dockerAPI.post/get/delete

### ❌ EXEC/SHELL FORBIDDEN ❌
- **NEVER** use `exec`, `execAsync`, `child_process`
- **NEVER** use shell commands for container management
- **ONLY** use Docker socket API for everything

### ❌ TIMEOUTS/SLEEP FORBIDDEN ❌
- **NEVER** use `sleep()` or arbitrary timeouts
- **ONLY** use port checks, log monitoring, or container status polling
- **ALWAYS** wait for actual readiness, not time-based delays

### ❌ DOCKERFILE FORBIDDEN ❌
- **NEVER** touch app/Dockerfile - it's working perfectly
- **KEEP CMD LINE** - working config has CMD ["python3", "pipeline_target.py"]
- **ONLY** PyYAML pip install, NVIDIA install.sh, pipeline_target.py copy
- **ENTRYPOINT-based setup** - files placed in example directory
- **WORKING CONFIGURATION** - don't break what works

### ✅ PROPER SEQUENCING (WAIT-FOR-IT) ✅
1. **Start MediaMTX** → wait for RTSP port 8554 ready
2. **Start FFmpeg sender** → wait for RTSP stream published in MediaMTX logs
3. **Start DeepStream** → wait for RTSP input connected + UDP output stream active
4. **Start recorder** → wait for 10 seconds of actual recording data
5. **Graceful shutdown** → kill entire sender/receiver chain via API

### 🏆 GOLDEN FLOW (CURRENT AUTOMATED TEST) 🏆
- **Command**: `npm test` - Live WebSocket streaming with real-time updates
- **Architecture**: WebSocket + API-driven Docker socket (no CLI, no timeouts)
- **Flow**: MediaMTX → FFmpeg → DeepStream → Recorder → Graceful shutdown
- **Output**: `golden_flow_*.mp4` in `/outputs/` with AI detection bounding boxes
- **Live Updates**: Real-time step progress via WebSocket connection

### 🚨 ANTI-RETARD GUIDANCE FOR FUTURE CLAUDE 🚨
**STOP BEING RETARDED:**
1. **USE CLEAN PIPELINE CODE**: No wait-for-streams, no timeouts, just pure Pipeline API
2. **FOLLOW EXACT FORMAT**: `from pyservicemaker import Pipeline` then `pipeline.start().wait()`
3. **USE WORKING DOCKERFILE**: hello_multi.py + source_multi.yaml + CMD structure
4. **DON'T ADD COMPLEXITY**: Keep it vanilla, let the golden flow handle sequencing
5. **TRUST THE WORKING CONFIG**: If Python script prints startup, it's working correctly
6. **CHECK ACTUAL LOGS**: Use `docker logs deepstream-master` to see real pipeline errors

### 🔧 NVMULTIURISRCBIN COMPLETE GUIDE 🔧
**ARCHITECTURE:**
- **Internal muxer** - no external nvstreammux needed
- **Direct linking** - nvmultiurisrcbin → nvinfer → nvosdbin → nvstreamdemux
- **Batch processing** - handles multiple streams internally

**CRITICAL PROPERTIES (CORRECTED):**
- `"uri-list": "rtsp://localhost:8554/cam1,rtsp://localhost:8554/cam2"` - comma-separated URIs (PORT 8554, NOT 8556!)
- `"sensor-id-list": "cam1,cam2"` - matching IDs
- `"sensor-name-list": "Camera1,Camera2"` - display names
- `"width": 1920, "height": 1080"` - resolution (NOT src-muxer-*)
- `"batch-size": 2` in nvinfer (must match number of streams)
- `"select-rtp-protocol": 4` - force TCP protocol to avoid UDP networking issues

**RTSP RESILIENCE:**
- `"select-rtp-protocol": 4` - force TCP (avoids UDP networking issues)
- `"rtsp-reconnect-interval": 10, "rtsp-reconnect-attempts": -1` - auto-reconnect
- `"latency": 100, "udp-buffer-size": 524288` - buffering

**REST API (Optional):**
- Default port 9000, can use `"port": 9000`
- If REST server fails, falls back to uri-list (this is OK)
- Dynamic stream addition via `/api/v1/stream/add` POST requests

**WORKING EXAMPLE FROM NVIDIA test5:**
```python
pipeline.add("nvmultiurisrcbin", "srcs", {
    "uri-list": "file:///path1.mp4,file:///path2.mp4",
    "sensor-id-list": "id1,id2",
    "sensor-name-list": "Name1,Name2",
    "width": 1920, "height": 1080,
    "batch-size": 2
}).link("srcs", "pgie")
```


## Project Overview
This project implements a dual-stream AI video processing pipeline using NVIDIA DeepStream 8.0 with PyServiceMaker, containerized with Docker and controlled via a Node.js web interface.

## Architecture Philosophy: Python + Node.js Dual Stack

### Why Two Languages? (The "Duo Poly" Approach)

**Python (DeepStream/AI Core)**
- **GPU Processing**: DeepStream SDK requires Python for PyServiceMaker API
- **AI/ML Ecosystem**: Native integration with NVIDIA's ML stack
- **Performance**: Direct GPU memory access, CUDA operations
- **Use Case**: Heavy computational tasks, batch processing, inference

**Node.js (Web Interface/Orchestration)**
- **Docker API**: Excellent ecosystem for container orchestration
- **Real-time**: WebSocket support, async I/O for streaming
- **Web Integration**: Native HTTP/REST APIs, frontend serving
- **Use Case**: User interface, API endpoints, container management

### Separation of Concerns

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │────│   Node.js API    │────│  Docker Socket  │
│   (Browser)     │    │   (Orchestrator)  │    │   (Container)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Python DeepStream│
                       │   (AI Pipeline)  │
                       └─────────────────┘
```

## Core Components

### 1. DeepStream Pipeline (Python)
**File**: `app/pipeline_target.py`
```python
# Target architecture: two_in_two_out
nvmultiurisrcbin → nvinfer (batch) → nvosdbin → nvstreamdemux → [rtspout0, rtspout1]
```

**Key Concepts**:
- **nvmultiurisrcbin**: Handles multiple RTSP inputs efficiently
- **Batch Processing**: Processes multiple streams together for GPU efficiency
- **Demux Pattern**: Splits processed streams back to individual outputs
- **Pipeline API**: Lower-level control vs Flow API (higher-level)

### 2. Web Interface (Node.js)
**File**: `webapp/server.js`
```javascript
// Docker API integration via unix socket
const dockerAPI = axios.create({
  socketPath: '/var/run/docker.sock',
  baseURL: 'http://localhost'
});
```

**Key Concepts**:
- **Docker Socket API**: Direct container control without shell exec
- **FFmpeg Containers**: Temporary containers for video processing
- **Host Networking**: Required for GPU access and RTSP streams
- **Stream Management**: Orchestrates video senders and recorders

## Development Context

### Current Status: WORKING PIPELINE
- ✅ nvmultiurisrcbin correctly configured with TCP protocol
- ✅ AI model building and inference working
- ✅ "new stream added" logs show streams being processed
- ✅ Single receiver loop processing dual streams
- ❌ Missing cam2 stream (404 Not Found) - need both streams for dual processing

### 📱 DUAL STREAM SETUP: FFmpeg + Larix iOS
**Perfect Configuration:**
- **cam1**: FFmpeg loop (consistent traffic video for AI testing)
- **cam2**: Larix iOS app (live mobile camera feed)

**Larix iOS Setup:**
1. Install Larix Broadcaster app on iOS
2. Set RTSP URL: `rtsp://YOUR_IP:8554/cam2`
3. Start publishing from iOS camera
4. MediaMTX will accept the stream automatically
5. nvmultiurisrcbin processes both cam1 + cam2 in single loop

**Benefits:**
- Real mobile video input for testing
- Dual-stream AI detection (file + live)
- Single receiver loop handles both streams
- Live AI bounding boxes on mobile feed

## Key Learning: Why Not Single Language?

### ❌ All Python Approach
- **Problem**: Web frameworks (Flask/FastAPI) add overhead
- **GPU Blocking**: Python web servers can block GPU operations
- **Container Complexity**: Mixing web + GPU in single container

### ❌ All Node.js Approach
- **Problem**: No native NVIDIA SDK bindings
- **Performance**: Cannot leverage GPU efficiently
- **Ecosystem**: Limited AI/ML library support

### ✅ Dual Stack Benefits
- **Specialization**: Each language handles what it does best
- **Isolation**: GPU workloads separated from web logic
- **Scalability**: Can scale web and AI components independently
- **Debugging**: Easier to troubleshoot separated concerns

## Development Workflow

### 1. DeepStream Development
```bash
# Build and test AI pipeline
cd /home/prafulrana/d
./build.sh && ./run.sh

# Monitor GPU usage
nvidia-smi

# Check RTSP outputs
ffplay rtsp://localhost:8554/out1
```

### 2. Web App Development
```bash
# Build and run web interface
cd /home/prafulrana/d/webapp
docker build -t deepstream-tester .
docker run -p 3001:3001 -v /var/run/docker.sock:/var/run/docker.sock deepstream-tester
```

### 3. Integration Testing
```bash
# Full stack test
1. Start DeepStream pipeline (Python)
2. Start web app (Node.js)
3. Use web interface to trigger FFmpeg containers
4. Verify AI detection in output videos
```

## Performance Considerations

### GPU Efficiency
- **Batch Processing**: Multiple streams processed together
- **Memory Management**: CUDA memory pools for efficiency
- **Pipeline Optimization**: Minimal data copies between GPU/CPU

### Container Overhead
- **Ephemeral Containers**: FFmpeg containers auto-remove after use
- **Resource Limits**: Set appropriate CPU/memory limits
- **Host Networking**: Eliminates bridge network overhead

### Real-time Streaming
- **RTSP Protocol**: Low-latency streaming for live applications
- **Buffer Management**: Proper buffering to prevent packet loss
- **Sync Parameters**: `sync=True` prevents network flooding

## Debugging Guide

### Common Issues

**1. GPU Not Accessible**
```bash
# Check GPU in container
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

**2. RTSP Streams Not Working**
```bash
# Test RTSP connectivity
ffplay rtsp://localhost:8554/stream-name

# Check port binding
netstat -tulpn | grep 8554
```

**3. Docker Socket Permissions**
```bash
# Web app can't access Docker
sudo chmod 666 /var/run/docker.sock
# Or add user to docker group
sudo usermod -a -G docker $USER
```

### Performance Monitoring
```bash
# GPU utilization
watch -n 1 nvidia-smi

# Container resource usage
docker stats

# Network traffic
sudo netstat -i
```

## Deployment Patterns

### Development
- Local GPU machine
- Direct Docker commands
- Hot reloading for web components

### Production
- Kubernetes with GPU operators
- Persistent volumes for model storage
- Load balancing for web tier
- Monitoring with Prometheus/Grafana

### Edge Deployment
- NVIDIA Jetson devices
- Lightweight web interface
- Local RTSP sources (cameras)
- Reduced model complexity for edge inference

## Security Considerations

### Docker Socket Access
- **Risk**: Full Docker API access from web app
- **Mitigation**: Use Docker-in-Docker or restricted socket proxy
- **Alternative**: Kubernetes with proper RBAC

### RTSP Streams
- **Authentication**: Add RTSP auth for production
- **Encryption**: Use RTSPS for sensitive streams
- **Network**: Isolate RTSP traffic in separate VLAN

### GPU Resources
- **Isolation**: Use NVIDIA MIG for multi-tenancy
- **Limits**: Set GPU memory limits per container
- **Monitoring**: Track GPU usage for abuse detection

---

## Production Deployment

### Ubuntu Service Setup
The web interface runs as a proper Ubuntu systemd service:

```bash
# Service management
sudo systemctl status deepstream-tester
sudo systemctl restart deepstream-tester
sudo systemctl logs -u deepstream-tester -f

# Service configuration
/etc/systemd/system/deepstream-tester.service
```

### Service Features:
- **Auto-start**: Enabled on boot
- **Docker Integration**: Direct Docker socket API access
- **Resource Limits**: 512MB memory limit
- **Security**: Runs as user with docker group access
- **Logging**: Centralized via journald

## Web Interface Features

### Test Workflow:
1. **User clicks "Test Stream X"** → 10-second spinner
2. **FFmpeg Sender**: Streams test video to DeepStream RTSP input
3. **AI Processing**: DeepStream processes with detection/classification
4. **FFmpeg Recorder**: Captures processed output for 10 seconds
5. **Video Playback**: Displays results with bounding boxes in browser

### iOS/Safari Compatibility:
- **H.264 Baseline Profile**: Safari-compatible encoding
- **Progressive Download**: `+faststart` for immediate playback
- **Range Requests**: Proper byte-range serving for iOS
- **CORS Headers**: Cross-origin streaming support

### Docker API Integration:
```javascript
// Direct Docker socket access (no shell exec)
const dockerAPI = axios.create({
  socketPath: '/var/run/docker.sock',
  baseURL: 'http://localhost'
});

// Container lifecycle management
await dockerAPI.post('/containers/create', config);
await dockerAPI.post(`/containers/${id}/start`);
```

## Quick Reference

**Start Everything:**
```bash
# 1. DeepStream AI Pipeline
cd /home/prafulrana/d && ./build.sh && ./run.sh

# 2. Web Interface (as service)
sudo systemctl start deepstream-tester

# 3. Access web UI
open http://localhost:3001
```

**Key Files:**
- `app/pipeline_target.py` - Main AI pipeline (Python)
- `webapp/server.js` - Web API server (Node.js)
- `webapp/public/index.html` - Frontend interface
- `/etc/systemd/system/deepstream-tester.service` - Ubuntu service
- `app/Dockerfile` - DeepStream container

**Ports:**
- 3001: Web interface
- 8554: RTSP output stream 1
- 8555: RTSP output stream 2
- 8556: RTSP input (MediaMTX)

## Testing Workflow

### Manual Testing:
```bash
# Test web API health
curl http://localhost:3001/api/health

# Test stream endpoint
curl -X POST http://localhost:3001/api/test-stream/0

# Check service logs
sudo journalctl -u deepstream-tester -f
```

### Browser Testing:
1. Navigate to `http://localhost:3001`
2. Click "Test Stream 0" or "Test Stream 1"
3. Wait 10 seconds for processing
4. Verify video playback with AI detection overlays

### Mobile Testing (iOS Safari):
- Videos use H.264 baseline profile for compatibility
- Progressive download with range request support
- Proper CORS headers for cross-origin access

---

## Current Implementation Plan

### Core Pipeline Structure (FINAL - DO NOT DEVIATE)
```python
from pyservicemaker import Pipeline

def main():
    pipeline = Pipeline("two_in_two_out")

    # Sources (two RTSP URIs)
    pipeline.add("nvmultiurisrcbin", "srcs", {
        "uri-list": "rtsp://localhost:8556/iphone-feed,rtsp://localhost:8556/iphone-feed"
    })

    # Batch mux
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 2,
        "width": 1280,
        "height": 720,
        "live-source": 1
    })

    # Inference
    pipeline.add("nvinfer", "pgie", {
        "config-file-path": "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"
    })

    # On-Screen Display
    pipeline.add("nvosdbin", "osd")

    # Demux
    pipeline.add("nvstreamdemux", "demux")

    # RTSP Outputs (per stream)
    pipeline.add("nvv4l2h264enc", "enc0", {"bitrate": 4000000})
    pipeline.add("rtph264pay", "pay0", {"pt": 96})
    pipeline.add("nvv4l2h264enc", "enc1", {"bitrate": 4000000})
    pipeline.add("rtph264pay", "pay1", {"pt": 96})

    # Linking
    pipeline.link("srcs", "mux", "pgie", "osd", "demux")
    pipeline.link("demux", "enc0", "pay0")
    pipeline.link("demux", "enc1", "pay1")

    pipeline.start().wait()

if __name__ == "__main__":
    main()
```

### Anti-Retard Measures

1. **ONLY ONE PYTHON FILE**: `app/pipeline_target.py` - all other Python files deleted
2. **EXACT DOCS STRUCTURE**: Uses PyServiceMaker Pipeline API exactly as documented
3. **NO DEVIATIONS**: Stick to nvmultiurisrcbin → nvstreammux → nvinfer → nvosdbin → nvstreamdemux pattern
4. **MediaMTX FOR RTSP**: Use bluenviron/mediamtx container for RTSP server (port 8556)
5. **FFMPEG FOR TESTING**: Web interface uses FFmpeg containers for senders/receivers only
6. **BASELINE MP4**: FFmpeg recorder uses `-pix_fmt yuv420p -profile:v baseline` for iOS compatibility

### Critical Success Path

1. **MediaMTX Running**: `docker run -d --name mediamtx --network host bluenviron/mediamtx:latest`
2. **Pipeline Container**: Uses exact structure above, nothing else
3. **Web Interface**: FFmpeg sender → MediaMTX → DeepStream → FFmpeg recorder → Baseline MP4
4. **AI Detection**: Traffic/person detection with bounding boxes in output video

### What NOT to Do

- ❌ Do not use Flow API
- ❌ Do not create multiple Python files
- ❌ Do not deviate from docs structure
- ❌ Do not use shell exec in web interface
- ❌ Do not use High profile MP4 encoding
- ❌ Do not bypass MediaMTX for RTSP

---

## Current Implementation Plan (Session Management)

### Persistent Container Strategy

**Goal**: Use docker.sock to manage persistent sessions. One FFmpeg feeder per stream that stays running. If already running, resume that session instead of creating new containers.

### Container Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MediaMTX      │    │   DeepStream     │    │  FFmpeg Feeders │
│   (Port 8556)   │◄───│   Pipeline       │◄───│  (Stream 0,1)   │
│   RTSP Server   │    │   (GPU AI)       │    │  (Persistent)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ FFmpeg Recorder │
                       │ (On-demand)     │
                       └─────────────────┘
```

### Session Management Logic

1. **Check Existing**: Use docker.sock to check if `ffmpeg-feeder-{streamId}` exists
2. **Resume or Create**:
   - If exists: Resume session (just start recorder)
   - If not exists: Create new persistent feeder
3. **DeepStream Pipeline**: Single persistent instance processing all streams
4. **Recorder**: On-demand per test, captures 10s, auto-removes

### Container Lifecycle

**Persistent Containers**:
- `mediamtx` - RTSP server (always running)
- `deepstream-pipeline` - AI processing (always running)
- `ffmpeg-feeder-0` - Stream 0 video source (persistent)
- `ffmpeg-feeder-1` - Stream 1 video source (persistent)

**Ephemeral Containers**:
- `ffmpeg-recorder-{streamId}` - 10s recording, auto-remove

### Implementation Steps

1. **Container State Management**: Check container status via docker.sock
2. **Feeder Persistence**: FFmpeg feeders loop indefinitely
3. **Resume Logic**: If feeder exists, skip creation and go to recording
4. **Health Checks**: Verify containers are actually streaming
5. **Error Recovery**: Restart failed containers automatically

### Anti-Retard Measures v2

- ✅ **Session Persistence**: Don't recreate what's already running
- ✅ **Resource Efficiency**: Reuse containers instead of constant churn
- ✅ **State Management**: Track container status properly
- ✅ **Docker.sock Only**: No shell exec, pure Docker API
- ✅ **Minimal Containers**: Only create what's needed when needed

---

## CRITICAL REQUIREMENT: VIDEO STREAMS NOT FILES

### Anti-Retard Rule #1: RTSP IN → AI PROCESSING → RTSP OUT

**The pipeline MUST process live video streams, NOT files.**

### Core Pipeline Structure (FINAL - EXACT DOCS FORMAT)
```python
from pyservicemaker import Pipeline

def main():
    pipeline = Pipeline("two_in_two_out")

    # --- Sources (two RTSP URIs) ---
    pipeline.add("nvmultiurisrcbin", "srcs", {
        "uris": [
            "rtsp://localhost:8556/cam1",
            "rtsp://localhost:8556/cam2"
        ]
    })

    # --- Batch mux ---
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 2,
        "width": 1280,
        "height": 720,
        "live-source": 1
    })

    # --- Inference ---
    pipeline.add("nvinfer", "pgie", {
        "config-file-path": "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"
    })

    # --- OSD (bin) ---
    pipeline.add("nvosdbin", "osd")

    # --- Demux ---
    pipeline.add("nvstreamdemux", "demux")

    # --- RTSP Outputs (per stream) ---
    pipeline.add("nvv4l2h264enc", "enc0", {"bitrate": 4000000})
    pipeline.add("rtph264pay", "pay0", {"pt": 96})
    pipeline.add("nvv4l2h264enc", "enc1", {"bitrate": 4000000})
    pipeline.add("rtph264pay", "pay1", {"pt": 96})

    # --- UDP Sinks for output ---
    pipeline.add("udpsink", "sink0", {"host": "224.224.255.255", "port": 5000})
    pipeline.add("udpsink", "sink1", {"host": "224.224.255.255", "port": 5001})

    # --- Linking ---
    pipeline.link("srcs", "mux", "pgie", "osd", "demux")

    # Link demux outputs explicitly (DOCS FORMAT)
    pipeline.link(("demux", "src_0"), "enc0", "pay0", "sink0")
    pipeline.link(("demux", "src_1"), "enc1", "pay1", "sink1")

    pipeline.start().wait()

if __name__ == "__main__":
    main()
```

### Video Stream Flow (ANTI-RETARD ARCHITECTURE)

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  FFmpeg Feeder  │────│    MediaMTX      │────│   DeepStream    │
│  (Live Video)   │    │  RTSP Server     │    │   AI Pipeline   │
│                 │    │  Port 8556       │    │   (GPU Process) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ FFmpeg Recorder │◄───│     UDP Out      │◄───│ RTSP Outputs    │
│ (Captures 10s)  │    │ Ports 5000/5001  │    │ With Bounding   │
│ Baseline MP4    │    │                  │    │ Boxes (AI)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### ANTI-RETARD CHECKLIST

**✅ MUST HAVE:**
1. **RTSP INPUT**: Live video streams from MediaMTX, NOT files
2. **AI PROCESSING**: Real-time inference with bounding boxes
3. **RTSP OUTPUT**: UDP streams that FFmpeg can record from
4. **NO FILE INPUTS**: Delete any file-based video sources

**❌ NEVER DO:**
- Use file inputs in DeepStream pipeline
- Skip MediaMTX for RTSP serving
- Output only to files without RTSP streams
- Create multiple Python pipeline files

### Video Requirements
- **INPUT**: RTSP streams from persistent FFmpeg feeders
- **PROCESSING**: GPU-accelerated AI detection with OSD
- **OUTPUT**: UDP streams for recording with AI bounding boxes
- **RESULT**: Baseline H.264 MP4 with visible AI detection results

---

## ANTI-RETARD TESTING MEASURES

### Testing Plan with Anti-Retard Mitigations

**GOAL**: Test the EXACT pipeline structure without ANY deviations or simplifications.

### 🚨 RETARD EXAMPLES TO PREVENT

**RETARD #1**: "Let me do simpler no DeepStream"
- **MITIGATION**: NO. DeepStream IS the core requirement. Test the actual pipeline or nothing.
- **PREVENTION**: Pipeline must use PyServiceMaker with GPU inference.

**RETARD #2**: "Let me do simpler no inference"
- **MITIGATION**: NO. AI inference IS the purpose. Without nvinfer, there's no point.
- **PREVENTION**: nvinfer element MUST be present and functional in all tests.

**RETARD #3**: "Let me read from file"
- **MITIGATION**: NO. Files defeat the RTSP streaming architecture.
- **PREVENTION**: ONLY RTSP inputs allowed. Check pipeline logs for file:// URIs.

**RETARD #4**: "Let me write a new file"
- **MITIGATION**: NO. We have ONE Python file: pipeline_target.py. That's it.
- **PREVENTION**: Delete any new .py files immediately. Only edit existing file.

**RETARD #5**: "Test pattern video is fine for testing"
- **MITIGATION**: NO. Test patterns have NO objects to detect. Completely useless.
- **PREVENTION**: MUST use traffic video with cars/people for AI detection to work.
- **REQUIREMENT**: Video content MUST contain detectable objects (vehicles, persons).

### Anti-Retard Testing Protocol

**PHASE 1: Infrastructure Verification**
```bash
# 1. MediaMTX running
docker ps | grep mediamtx || FAIL

# 2. DeepStream container builds
cd /home/prafulrana/d && ./build.sh || FAIL

# 3. Pipeline starts without errors
docker run --gpus all --network host pyservicemaker-hello:latest || FAIL
```

**PHASE 2: Stream Flow Verification**
```bash
# 1. FFmpeg feeder streams to MediaMTX
docker logs ffmpeg-feeder-0 | grep "Opening" || FAIL

# 2. DeepStream connects to RTSP
docker logs deepstream-pipeline | grep "new stream added" || FAIL

# 3. UDP output streams active
netstat -tulpn | grep "5000\|5001" || FAIL
```

**PHASE 3: AI Detection Verification**
```bash
# 1. Test generates baseline MP4
ls -la /tmp/stream_0_output.mp4 || FAIL

# 2. MP4 has baseline profile
ffprobe /tmp/stream_0_output.mp4 | grep "Constrained Baseline" || FAIL

# 3. Video plays with bounding boxes (manual verification)
```

### Anti-Retard Enforcement Rules

**🔒 PIPELINE IMMUTABILITY**
- pipeline_target.py is THE ONLY Python file
- EXACT structure from docs - no modifications
- nvmultiurisrcbin → nvstreammux → nvinfer → nvosdbin → nvstreamdemux
- Tuple linking: `("demux", "src_0")` format required

**🔒 RTSP-ONLY ARCHITECTURE**
- NO file inputs: no sample_1080p_h264.mp4 in DeepStream
- RTSP inputs: cam1/cam2 streams from MediaMTX
- UDP outputs: ports 5000/5001 for recording

**🔒 NO SIMPLIFICATION ALLOWED**
- DeepStream MUST be used (no simple FFmpeg-only)
- AI inference MUST be active (no passthrough)
- GPU processing MUST be working (check nvidia-smi)
- Full pipeline MUST be tested (no shortcuts)

### Failure Responses to Retard Suggestions

**"Can we skip DeepStream for now?"** → NO. Test the real system.
**"Let's just use file input first"** → NO. RTSP streams only.
**"Maybe remove AI to test basics"** → NO. AI is the core purpose.
**"Should I create a simple test file?"** → NO. Use pipeline_target.py only.
**"Can we simplify the linking?"** → NO. Use exact docs tuple format.

### Success Criteria (All Must Pass)

✅ DeepStream pipeline starts and runs continuously
✅ RTSP inputs connect from MediaMTX (cam1/cam2)
✅ AI inference processes video with bounding boxes
✅ UDP outputs stream to ports 5000/5001
✅ FFmpeg recorder captures baseline MP4
✅ Output video shows AI detection results
✅ No file inputs anywhere in the pipeline
✅ No additional Python files created
✅ Exact PyServiceMaker docs structure maintained

### 📸 ANTI-RETARD PROOF REQUIREMENT

**CRITICAL**: Must provide SCREENSHOT showing:
- MP4 video playing in browser
- Visible AI detection bounding boxes around objects
- Timestamp showing real processing (not static image)

**WHERE TO SCREENSHOT**:
- Browser showing `http://localhost:3001/api/stream/0`
- Video player with visible detection boxes
- Clear evidence of AI processing in action

**ANTI-RETARD RULE**: No claims of "working AI detection" without visual proof. Screenshots must show actual bounding boxes on detected objects (cars, people, etc.) overlaid on the video feed.

**FAILURE CASES TO PREVENT**:
- ❌ Video plays but no bounding boxes visible
- ❌ Static test pattern without real AI detection
- ❌ File playback claiming to be "live detection"
- ❌ Blank/black video claiming "AI is working"
- ❌ **TEST PATTERN VIDEO** - Geometric patterns have NO objects to detect!

**SCREENSHOT MUST SHOW**:
- **REAL TRAFFIC VIDEO**: Cars, people, moving objects
- **AI DETECTION BOXES**: Colorful bounding boxes with labels (Vehicle, Person, etc.)
- **NOT TEST PATTERNS**: Geometric shapes cannot be detected by AI

### 🚨 VIDEO CONTENT ANTI-RETARD RULE

**CRITICAL**: The DeepStream AI model is trained for traffic detection. Test patterns are USELESS.

**REQUIRED VIDEO CONTENT**:
- Traffic scenes with vehicles
- People walking/crossing
- Real-world objects the AI can detect
- Moving content (not static images)

**FORBIDDEN VIDEO CONTENT**:
- Geometric test patterns
- Color bars
- Static test images
- Abstract patterns

**FFmpeg Feeder MUST stream traffic video content, NOT test patterns!**

---

## SERVICE RESILIENCE ANTI-RETARD PLAN

### Problem: Intermittent RTSP Inputs

The nvmultiurisrcbin element fails with "Output width not set" when RTSP streams are unavailable during pipeline startup. This breaks the entire AI processing pipeline.

### Root Cause Analysis

1. **nvmultiurisrcbin requires active streams**: Cannot determine stream properties without connected sources
2. **Pipeline initialization order**: nvstreammux needs width/height before nvmultiurisrcbin can connect
3. **No graceful degradation**: Pipeline fails completely instead of waiting for streams
4. **Container restart loops**: Failed containers restart infinitely without fixing root cause

### Resilience Strategy

**PHASE 1: RTSP Stream Resilience Configuration**

Update pipeline_target.py with nvmultiurisrcbin resilience settings:

```python
pipeline.add("nvmultiurisrcbin", "srcs", {
    "uri-list": "rtsp://localhost:8556/cam1,rtsp://localhost:8556/cam2",
    "rtsp-reconnect-interval": 5,      # Reconnect after 5s timeout
    "rtsp-reconnect-attempts": -1,     # Infinite reconnection attempts
    "drop-pipeline-eos": True,         # Don't terminate on stream end
    "async-handling": True,            # Handle state changes async
    "low-latency-mode": True          # Optimize for IPPP frames
})
```

**PHASE 2: Service Dependency Management**

Container startup order with health checks:

```yaml
# docker-compose-like dependency structure
1. MediaMTX (RTSP server) - ALWAYS FIRST
2. FFmpeg feeders - WAIT for MediaMTX health
3. DeepStream pipeline - WAIT for RTSP streams available
4. Web interface - WAIT for DeepStream ready
```

**PHASE 3: Stream Availability Detection**

Pre-flight checks before pipeline start:

```bash
# Check RTSP stream availability
curl -f "rtsp://localhost:8556/cam1" --max-time 3 || WAIT
curl -f "rtsp://localhost:8556/cam2" --max-time 3 || WAIT

# Verify stream properties
ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height rtsp://localhost:8556/cam1
```

**PHASE 4: Graceful Degradation**

Pipeline behavior when streams unavailable:

1. **Startup Mode**: Wait up to 30s for streams before failing
2. **Runtime Mode**: Continue processing available streams if one fails
3. **Recovery Mode**: Automatically reconnect when streams return
4. **Fallback Mode**: Use dummy source if all streams fail (development only)

**PHASE 5: Container Health Monitoring**

Web service health checks with auto-recovery:

```javascript
// Health check intervals
setInterval(checkContainerHealth, 10000);  // Every 10s

async function checkContainerHealth() {
  const containers = ['mediamtx', 'deepstream-pipeline'];
  for (const name of containers) {
    const status = await checkContainerStatus(name);
    if (!status.running) {
      await restartContainer(name);
    }
  }
}
```

### Anti-Retard Failure Prevention

**RETARD BEHAVIOR #1**: "Just restart everything when it fails"
- **MITIGATION**: NO. Understand WHY it failed and fix root cause
- **PREVENTION**: Implement proper dependency ordering and health checks

**RETARD BEHAVIOR #2**: "Skip resilience, fix later"
- **MITIGATION**: NO. Resilience is core requirement for production
- **PREVENTION**: Build resilience into initial implementation, not retrofitted

**RETARD BEHAVIOR #3**: "Use file inputs when RTSP fails"
- **MITIGATION**: NO. Files defeat the live streaming architecture
- **PREVENTION**: Wait for RTSP or fail gracefully, never fall back to files

**RETARD BEHAVIOR #4**: "Simplify by removing nvmultiurisrcbin"
- **MITIGATION**: NO. This element is core to multi-stream efficiency
- **PREVENTION**: Fix the configuration, don't bypass the architecture

### Implementation Priority

1. **IMMEDIATE**: Add resilience config to nvmultiurisrcbin
2. **HIGH**: Implement stream availability checks
3. **MEDIUM**: Add container health monitoring
4. **LOW**: Graceful degradation modes

### Success Metrics

✅ Pipeline starts successfully even when RTSP streams delayed
✅ Automatic reconnection when streams become available
✅ No container restart loops due to missing streams
✅ Graceful handling of single stream failures
✅ Health monitoring with auto-recovery
✅ Zero manual intervention required for stream interruptions

### Testing Protocol

**Resilience Test Cases**:
1. Start pipeline before RTSP streams available
2. Disconnect one stream during processing
3. Disconnect both streams and reconnect
4. Stop/start MediaMTX while pipeline running
5. Network interruption simulation

**Expected Behavior**:
- Pipeline waits for streams gracefully
- Automatic reconnection without restart
- Continued processing of available streams
- Recovery without data loss
- No container restart loops

---

## MASTER STARTUP/RESTART ANTI-RETARD PLAN

### Problem: Manual Container Management Hell

Currently we're manually starting containers with docker CLI commands. This is retarded for production. Need master orchestration via docker.sock API only.

### Master Service Architecture

**SINGLE UBUNTU SERVICE** controls everything via Docker API:

```
┌─────────────────────────────────────────────────┐
│           deepstream-tester.service             │
│                 (Master)                        │
├─────────────────────────────────────────────────┤
│  1. Cleanup existing containers                 │
│  2. Start MediaMTX (RTSP server)               │
│  3. Start FFmpeg feeders (traffic video)       │
│  4. Start DeepStream pipeline (AI)             │
│  5. Serve web interface                        │
│  6. Health monitoring & auto-restart           │
└─────────────────────────────────────────────────┘
```

### Master Startup Sequence (ANTI-RETARD)

**PHASE 1: CLEANUP**
```javascript
// Remove ALL existing containers (nuclear option)
const containers = await dockerAPI.get('/containers/json', { params: { all: true } });
for (const container of containers.data) {
  if (container.Names.some(name =>
    name.includes('mediamtx') ||
    name.includes('ffmpeg') ||
    name.includes('deepstream')
  )) {
    await dockerAPI.post(`/containers/${container.Id}/kill`);
    await dockerAPI.delete(`/containers/${container.Id}`);
  }
}
```

**PHASE 2: SEQUENTIAL STARTUP WITH HEALTH CHECKS**
```javascript
async function masterStartup() {
  console.log("🚀 MASTER STARTUP SEQUENCE");

  // 1. MediaMTX (RTSP Server)
  await startMediaMTX();
  await waitForHealthy('mediamtx', 'http://localhost:9997/v3/config');

  // 2. FFmpeg Feeders (Traffic Video)
  await startFFmpegFeeders();
  await waitForRTSPStreams(['rtsp://localhost:8556/cam1', 'rtsp://localhost:8556/cam2']);

  // 3. DeepStream Pipeline (AI Processing)
  await startDeepStreamPipeline();
  await waitForUDPOutputs([5000, 5001]);

  console.log("✅ ALL SERVICES READY");
}
```

**PHASE 3: CONTAINER DEFINITIONS**
```javascript
const CONTAINER_CONFIGS = {
  mediamtx: {
    Image: 'bluenviron/mediamtx:latest',
    name: 'mediamtx-master',
    HostConfig: {
      NetworkMode: 'host',
      RestartPolicy: { Name: 'unless-stopped' }
    },
    ExposedPorts: {
      '8554/tcp': {},
      '8556/tcp': {},
      '9997/tcp': {}
    }
  },

  ffmpegCam1: {
    Image: 'jrottenberg/ffmpeg:4.1-alpine',
    name: 'ffmpeg-cam1-master',
    Cmd: [
      '-re', '-stream_loop', '-1',
      '-i', '/samples/sample_1080p_h264.mp4',
      '-c', 'copy', '-f', 'rtsp',
      'rtsp://localhost:8556/cam1'
    ],
    HostConfig: {
      NetworkMode: 'host',
      Binds: ['/opt/nvidia/deepstream/deepstream/samples/streams:/samples:ro'],
      RestartPolicy: { Name: 'unless-stopped' }
    }
  },

  ffmpegCam2: {
    Image: 'jrottenberg/ffmpeg:4.1-alpine',
    name: 'ffmpeg-cam2-master',
    Cmd: [
      '-re', '-stream_loop', '-1',
      '-i', '/samples/sample_1080p_h264.mp4',
      '-c', 'copy', '-f', 'rtsp',
      'rtsp://localhost:8556/cam2'
    ],
    HostConfig: {
      NetworkMode: 'host',
      Binds: ['/opt/nvidia/deepstream/deepstream/samples/streams:/samples:ro'],
      RestartPolicy: { Name: 'unless-stopped' }
    }
  },

  deepstream: {
    Image: 'pyservicemaker-hello:latest',
    name: 'deepstream-master',
    HostConfig: {
      NetworkMode: 'host',
      DeviceRequests: [{ Driver: 'nvidia', Count: -1, Capabilities: [['gpu']] }],
      RestartPolicy: { Name: 'unless-stopped' }
    }
  }
};
```

**PHASE 4: HEALTH CHECK FUNCTIONS**
```javascript
async function waitForHealthy(containerName, healthUrl, timeout = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try {
      if (healthUrl) {
        await axios.get(healthUrl, { timeout: 2000 });
      }
      const container = await getContainerStatus(containerName);
      if (container.running) {
        console.log(`✅ ${containerName} healthy`);
        return true;
      }
    } catch (error) {
      console.log(`⏳ Waiting for ${containerName}...`);
    }
    await sleep(2000);
  }
  throw new Error(`❌ ${containerName} failed health check`);
}

async function waitForRTSPStreams(streams, timeout = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    let allReady = true;
    for (const stream of streams) {
      try {
        const result = await execAsync(`ffprobe -v quiet -timeout 3000000 "${stream}"`);
        if (result.code !== 0) allReady = false;
      } catch {
        allReady = false;
      }
    }
    if (allReady) {
      console.log(`✅ All RTSP streams ready`);
      return true;
    }
    await sleep(2000);
  }
  throw new Error(`❌ RTSP streams not ready after ${timeout}ms`);
}
```

### Anti-Retard Master Service Integration

**UPDATE server.js** with master startup:

```javascript
// On service start
process.on('SIGTERM', cleanup);
process.on('SIGINT', cleanup);

async function cleanup() {
  console.log("🧹 CLEANUP: Stopping all managed containers");
  await cleanupContainers();
  process.exit(0);
}

// Start master sequence
masterStartup().catch(error => {
  console.error("💀 MASTER STARTUP FAILED:", error);
  process.exit(1);
});

app.listen(PORT, () => {
  console.log(`🎯 DeepStream Master Controller running on port ${PORT}`);
});
```

### Systemd Service Updates

**NO DOCKER CLI DEPENDENCIES**:
```ini
[Unit]
Description=DeepStream Master Controller
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=prafulrana
Group=docker
WorkingDirectory=/home/prafulrana/d/webapp
Environment=NODE_ENV=production
Environment=PORT=3001
ExecStart=/usr/bin/node server.js
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
Restart=always
RestartSec=10

# Docker socket access only
SupplementaryGroups=docker

# Resource limits
LimitNOFILE=65536
MemoryMax=512M

[Install]
WantedBy=multi-user.target
```

### Anti-Retard Rules for Master Service

**✅ MUST DO:**
1. **Docker API ONLY**: No shell exec, no docker CLI commands
2. **Sequential Startup**: Wait for each service before starting next
3. **Health Checks**: Verify each service is actually working
4. **Nuclear Cleanup**: Kill everything before starting (idempotent)
5. **Auto-Restart**: Service restarts fix all container issues

**❌ NEVER DO:**
- Manual docker run commands
- Shell exec for container management (execAsync, child_process)
- Starting containers without health checks
- Partial cleanup (all-or-nothing)
- Multiple startup scripts
- **CRITICAL**: `exec` or `execAsync` in web interface - always use Docker socket API
- **CRITICAL**: No timeouts/sleep - use port checks or log monitoring only

### Master Service Benefits

🎯 **Single Point of Control**: One systemd service manages everything
🧹 **Idempotent**: Can restart anytime, always gets clean state
🔄 **Self-Healing**: Auto-restart fixes container issues
📊 **Health Monitoring**: Knows when each component is actually ready
🚀 **Production Ready**: No manual intervention required

### Implementation Steps

1. **Update server.js** with master startup sequence
2. **Add health check functions** for each service type
3. **Test nuclear cleanup** and full restart cycle
4. **Verify systemd integration** with proper signal handling
5. **Validate traffic video** flows through complete pipeline

### ON-DEMAND CONTAINER ARCHITECTURE

**CRITICAL UPDATE**: FFmpeg containers start on-demand via axios/docker.sock API, NOT at service startup.

**Master Service Startup (Minimal)**:
```
1. MediaMTX (RTSP server) - Always running
2. Web interface ready
3. FFmpeg feeders - START ON-DEMAND
4. DeepStream pipeline - START ON-DEMAND
```

**On-Demand Flow via Docker.sock API**:
```javascript
// User clicks "Test Stream 0"
app.post('/api/test-stream/0', async (req, res) => {
  // 1. Start FFmpeg feeder via axios
  await dockerAPI.post('/containers/create', ffmpegCam1Config);
  await dockerAPI.post(`/containers/${id}/start`);

  // 2. Wait for RTSP stream ready
  await waitForRTSPStreams(['rtsp://localhost:8556/cam1']);

  // 3. Start DeepStream if not running
  await dockerAPI.post('/containers/create', deepstreamConfig);
  await dockerAPI.post(`/containers/${id}/start`);

  // 4. Start recorder
  await dockerAPI.post('/containers/create', recorderConfig);
  await dockerAPI.post(`/containers/${id}/start`);
});
```

**Anti-Retard Benefits**:
- ✅ **NO DOCKER CLI**: All container management via axios API calls
- ✅ **ON-DEMAND**: Resources only used when testing
- ✅ **RESILIENT**: Containers start only when needed
- ✅ **ATOMIC**: Each test is independent container lifecycle
- ✅ **PRODUCTION**: No manual intervention required

**Container Lifecycle**:
- **MediaMTX**: Persistent (always running)
- **FFmpeg Feeders**: On-demand (per test)
- **DeepStream**: On-demand (shared across tests)
- **Recorders**: Ephemeral (10s auto-remove)

---

## SIMPLIFIED RESTART ANTI-RETARD APPROACH

### Problem: Over-Engineering Master Sequences

The master startup sequence is unnecessarily complex. Keep it simple.

### ANTI-RETARD SIMPLE APPROACH

**KISS Principle**: Ubuntu service runs Node.js, containers start on-demand only.

**Service Startup (MINIMAL)**:
```bash
# Ubuntu service starts
sudo systemctl start deepstream-tester

# Node.js starts
node server.js

# MediaMTX starts immediately (only persistent container)
# Everything else: ON-DEMAND via axios/docker.sock
```

**Restart Protocol (NUCLEAR OPTION)**:
```bash
# Full system restart
sudo systemctl restart docker      # Restart Docker daemon (kills all containers)
sudo systemctl restart deepstream-tester  # Restart Node.js service
```

**Benefits of Simple Approach**:
- ✅ **NO COMPLEX STARTUP**: Just start Node.js
- ✅ **DOCKER RESTART**: Nuclear option kills all containers
- ✅ **ON-DEMAND ONLY**: Containers created when needed
- ✅ **STATELESS**: Every restart is clean slate
- ✅ **PRODUCTION READY**: systemctl handles everything

### Anti-Retard Rules (SIMPLIFIED)

**✅ DO:**
1. **Ubuntu service runs Node.js** - That's it
2. **Docker restart for cleanup** - Nuclear option works
3. **On-demand containers** - Start only when testing
4. **axios/docker.sock** - No CLI commands ever

**❌ DON'T:**
- Complex startup sequences
- Master orchestration logic
- Pre-starting containers
- Health check loops at startup
- Dependency management complexity

### Implementation (KEEP IT SIMPLE)

**server.js startup**:
```javascript
// Just start the web server
app.listen(PORT, () => {
  console.log(`DeepStream Tester running on port ${PORT}`);
});

// Containers start on-demand in /api/test-stream/:id
```

**Restart workflow**:
```bash
# When things go wrong
sudo systemctl restart docker
sudo systemctl restart deepstream-tester
# Done. Everything clean.
```

**Why This Works**:
- Docker restart kills all containers (nuclear cleanup)
- Node.js service restart gives clean process
- On-demand containers ensure no orphans
- Simple = reliable = production ready

---

## MEDIAMTX ANTI-RETARD CONFIGURATION

### Problem: MediaMTX RTSP Publishing Configuration Hell

MediaMTX needs proper configuration to accept RTSP publishing from FFmpeg. Default configuration often fails.

### ANTI-RETARD SOLUTION: KNOWN WORKING CONFIG

**MediaMTX Default Behavior**:
- RTSP server listens on port 8554 (output/playback)
- RTSP publishing accepts streams on same port 8554
- **NOT 8556** - that's a red herring

**Correct FFmpeg Command** (ANTI-RETARD):
```bash
# WRONG (what we were doing)
ffmpeg -re -i video.mp4 -c copy -f rtsp rtsp://localhost:8556/cam1

# RIGHT (what actually works)
ffmpeg -re -i video.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1
```

**MediaMTX Container Config** (ANTI-RETARD):
```javascript
const MEDIAMTX_CONFIG = {
  Image: 'bluenviron/mediamtx:latest',
  name: 'mediamtx-master',
  HostConfig: {
    NetworkMode: 'host',
    RestartPolicy: { Name: 'unless-stopped' }
  },
  ExposedPorts: {
    '8554/tcp': {},  // RTSP server (publish AND playback)
    '1935/tcp': {},  // RTMP (optional)
    '8888/tcp': {}   // HLS (optional)
  }
};
```

**FFmpeg Feeder Config** (ANTI-RETARD):
```javascript
const FFMPEG_FEEDER_CONFIG = {
  Image: 'jrottenberg/ffmpeg:4.1-alpine',
  Cmd: [
    '-re', '-stream_loop', '-1',
    '-i', '/samples/sample_1080p_h264.mp4',
    '-c', 'copy', '-f', 'rtsp',
    'rtsp://localhost:8554/cam1'  // Port 8554, NOT 8556!
  ],
  HostConfig: {
    NetworkMode: 'host',
    Binds: ['/opt/nvidia/deepstream/deepstream/samples/streams:/samples:ro']
  }
};
```

**DeepStream Pipeline Config** (ANTI-RETARD):
```python
pipeline.add("nvmultiurisrcbin", "srcs", {
    "uri-list": "rtsp://localhost:8554/cam1,rtsp://localhost:8554/cam2"  # Port 8554!
})
```

### Anti-Retard MediaMTX Rules

**✅ DO:**
1. **Use port 8554** for both publishing and playback
2. **Default MediaMTX config** works out of the box
3. **Host networking** for container
4. **Test with ffplay** before pipeline: `ffplay rtsp://localhost:8554/cam1`

**❌ DON'T:**
- Use port 8556 (random wrong port)
- Override MediaMTX configuration unnecessarily
- Use separate ports for publish/playback
- Bridge networking (breaks RTSP)

### Testing Protocol (ANTI-RETARD)

**Step 1: Start MediaMTX**
```bash
docker run -d --name mediamtx --network host bluenviron/mediamtx:latest
```

**Step 2: Publish Stream**
```bash
docker run --rm --network host -v /path/to/video:/video:ro jrottenberg/ffmpeg:4.1-alpine \
  -re -i /video/sample.mp4 -c copy -f rtsp rtsp://localhost:8554/test
```

**Step 3: Verify Stream**
```bash
ffplay rtsp://localhost:8554/test
# Should show video playing
```

**Step 4: Check Available Streams**
```bash
curl http://localhost:9997/v3/paths
# Should show "test" path with 1 reader
```

### Common MediaMTX Failures (ANTI-RETARD)

**FAILURE: "Connection refused"**
- CAUSE: Using wrong port (8556 instead of 8554)
- FIX: Use port 8554 for everything

**FAILURE: "Stream not found"**
- CAUSE: FFmpeg not publishing or using wrong stream name
- FIX: Check stream name consistency (cam1, cam2, etc.)

**FAILURE: "Network unreachable"**
- CAUSE: Bridge networking instead of host
- FIX: Use `--network host` for all containers

**FAILURE: "Stream disconnects"**
- CAUSE: FFmpeg container auto-removing too quickly
- FIX: Use `-stream_loop -1` for continuous streaming

---

## DEEPSTREAM VERSION ANTI-RETARD LOCKDOWN

### Problem: Container Version Chaos

Docker keeps downloading different DeepStream versions instead of using the existing one. This wastes time and bandwidth.

### ANTI-RETARD SOLUTION: VERSION LOCKDOWN

**WE HAVE DEEPSTREAM 8.0 MULTIARCH - STOP DOWNLOADING OTHERS**

**Container Image Status Check**:
```bash
# Check what we have
docker images | grep deepstream
docker images | grep nvcr.io/nvidia/deepstream

# Should show: nvcr.io/nvidia/deepstream:8.0-triton-multiarch
```

**Dockerfile Anti-Retard Lockdown**:
```dockerfile
# LOCKED VERSION - DO NOT CHANGE
FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# If this image exists locally, Docker will NOT download again
# Only downloads if:
# 1. Image doesn't exist locally
# 2. Tag changes (8.0 -> 8.1)
# 3. Image explicitly pulled with --pull=always
```

**Anti-Retard Container Rules**:

**✅ DO:**
1. **Use exact tag**: `8.0-triton-multiarch` (what we have)
2. **Check existing images** before build
3. **Reuse existing containers** when possible
4. **Pin versions** in production

**❌ NEVER DO:**
- Use `latest` tag (downloads every time)
- Use different DeepStream versions (8.1, 7.x, etc.)
- Force pull with `--pull=always` unless necessary
- Mix DeepStream versions in same project

### Version Verification (ANTI-RETARD)

**Before Building**:
```bash
# Check if we already have the image
docker images nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# If present, build will be FAST (cache hit)
# If missing, will download ~8GB (slow)
```

**During Build**:
```bash
# Should see "Using cache" for FROM step
Step 1/5 : FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch
 ---> e6c06cd181ca  # <-- This means using cached image
```

**Container Size Check**:
```bash
# DeepStream 8.0 base image should be ~8GB
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep deepstream
```

### Anti-Retard Build Optimization

**Fast Build (Cache Hit)**:
```
Step 1/5 : FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch
 ---> e6c06cd181ca   # Using cache - GOOD
```

**Slow Build (Download)**:
```
Step 1/5 : FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch
8.0-triton-multiarch: Pulling from nvidia/deepstream  # BAD - downloading
```

### Emergency Version Reset

**If Wrong Version Downloaded**:
```bash
# Remove wrong versions
docker rmi $(docker images | grep deepstream | grep -v 8.0-triton-multiarch | awk '{print $3}')

# Verify only correct version remains
docker images | grep deepstream
# Should only show: 8.0-triton-multiarch
```

### Production Version Control

**Container Registry Strategy**:
- ✅ **Tag with specific version**: `myregistry/deepstream:8.0-multiarch-v1`
- ✅ **Pin exact SHA**: `nvcr.io/nvidia/deepstream@sha256:abc123...`
- ❌ **Never use latest**: Downloads unpredictably
- ❌ **Never mix versions**: Causes confusion

### Anti-Retard Verification

**Check Current Status**:
```bash
# What DeepStream images do we have?
docker images | grep -E "(deepstream|nvcr.io/nvidia)"

# What's our build using?
grep "FROM" /home/prafulrana/d/app/Dockerfile
# Should be: FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Are we accidentally pulling?
grep -r "docker pull" /home/prafulrana/d/
# Should be empty (no manual pulls)
```

**Success Criteria**:
- ✅ Only one DeepStream version present: `8.0-triton-multiarch`
- ✅ Dockerfile uses exact version tag
- ✅ Build shows "Using cache" for FROM step
- ✅ No accidental downloads during builds

---

## ANTI-RETARD: NO SERVICE RESTARTS FOR CONFIG CHANGES

### Problem: Restarting Service for Simple Changes

We keep doing `systemctl restart deepstream-tester` for config changes. This is RETARDED.

### ANTI-RETARD RULE: API-ONLY APPROACH

**The Node.js service should run continuously. All container management via Docker API only.**

**❌ NEVER DO:**
```bash
# WRONG - restarting service for config changes
sudo systemctl restart deepstream-tester

# WRONG - restarting Docker daemon
sudo systemctl restart docker
```

**✅ CORRECT API APPROACH:**
```bash
# Service runs continuously
# Containers managed on-demand via docker.sock API
curl -X POST http://localhost:3001/api/test-stream/0
```

**Why API-Only Works:**
- ✅ **Hot reload**: Config changes take effect immediately
- ✅ **Zero downtime**: Service stays running
- ✅ **Container lifecycle**: Managed via docker.sock
- ✅ **Stateless**: Each test is independent
- ✅ **Production ready**: No service interruptions

**API-Driven Container Management:**
```javascript
// All via axios/docker.sock - NO systemctl commands
await dockerAPI.post('/containers/create', config);
await dockerAPI.post(`/containers/${id}/start`);
await dockerAPI.post(`/containers/${id}/kill`);
await dockerAPI.delete(`/containers/${id}`);
```

**When to Restart Service (ONLY):**
- ❌ Config changes (use API)
- ❌ Container issues (use API)
- ❌ Port changes (use API)
- ✅ **ONLY for major Node.js code changes**

### Anti-Retard Service Rules

**✅ DO:**
1. **Keep service running** - continuous operation
2. **Use docker.sock API** - for all container operations
3. **Hot reload configs** - via API calls
4. **Independent tests** - each API call is self-contained

**❌ DON'T:**
- Restart service for config changes
- Restart Docker daemon unnecessarily
- Use systemctl for container management
- Mix CLI and API approaches

### CORRECT RESTART PROTOCOL (ANTI-RETARD)

**npm start = restart docker daemon + ubuntu nodejs service**

```bash
# The ONLY restart command needed
npm start

# Which does:
# 1. sudo systemctl restart docker      (nuclear cleanup - kills all containers)
# 2. sudo systemctl restart deepstream-tester  (restart Node.js service)
```

**Why This Works:**
- ✅ **Nuclear cleanup**: Docker restart kills ALL containers
- ✅ **Fresh start**: Node.js service gets clean slate
- ✅ **One command**: Simple, predictable, repeatable
- ✅ **Production ready**: Systemctl handles everything
- ✅ **Idempotent**: Can run anytime, always works

**package.json Anti-Retard Script:**
```json
{
  "scripts": {
    "start": "sudo systemctl restart docker && sudo systemctl restart deepstream-tester"
  }
}
```

**Usage:**
```bash
# When anything goes wrong or config changes
npm start

# Then test immediately
curl -X POST http://localhost:3001/api/test-stream/0
```