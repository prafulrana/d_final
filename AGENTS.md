# Repository Guidelines

This repo hosts a **Python-based single-stream DeepStream pipeline** that processes RTSP input with inference and outputs to RTSP via rtspclientsink. It uses nvurisrcbin for automatic RTSP reconnection.

## Project Structure & Modules
- `app.py` — Main Python application: nvurisrcbin → nvstreammux → nvinfer → OSD → encoder → rtspclientsink
- `Dockerfile` — DeepStream 8.0 Python environment with TensorRT engine caching
- `up.sh` — Multi-pipeline orchestration script (runs 6 concurrent inference containers)
- `models/` — Persistent TensorRT engine cache (volume mounted)
- `config/` — nvinfer configurations for different models (TrafficCamNet, PeopleNet, Segmentation, etc.)
- `relay/` — MediaMTX relay server configuration (GCP VM). Default zone: `asia-south1-c`
- `STANDARDS.md`, `STRUCTURE.md` — Build/test/debug documentation

## Build, Test, Run
- Build & Run: `./up.sh` (builds `ds_python:latest`, runs 6 concurrent pipelines: `drishti-s0` through `drishti-s5`)
- View logs: `docker logs -f drishti-s0` (or s1, s2, s3, s4, s5)
- Monitor all: `docker ps --filter ancestor=ds_python:latest`
- Publish test stream: Use Larix app or `gst-launch-1.0` to `rtsp://server:8554/in_s0`
- View outputs: `http://server:8889/s0/` (or s1, s2, s3, s4, s5 via WebRTC)
- Relay deploy: `cd relay && terraform init && terraform apply -var project_id=<GCP_PROJECT>`

## Coding Style & Conventions
- Python 3 with GStreamer bindings (`gi.repository.Gst`)
- 4-space indentation, max 100 character lines
- Follow NVIDIA DeepStream Python sample patterns
- Use `sys.stderr.write()` for errors, `print()` for info
- Keep functions focused with early returns

## Testing Guidelines
- **Initial connection test**: Publish to `in_s0`, verify outputs at `s0`-`s5`, check logs for "Pipeline is PLAYING"
- **Reconnection test**: Stop publisher, wait ~10s for reconnect logs, restart publisher, verify auto-recovery
- **Video quality test**: Compare smoothness of `in_s0` vs all outputs - outputs should match input quality
- **Performance test**: Verify startup time <10s with cached TensorRT engine (first pipeline), concurrent startup for others
- **Multi-pipeline test**: Verify all 6 pipelines run concurrently without GPU exhaustion (proven to handle 168+ streams)
- Include logs and minimal repro for any pipeline or encoder changes

## Multi-Pipeline Architecture
- **Input**: Single RTSP stream at `rtsp://relay:8554/in_s0`
- **Output**: 6 concurrent RTSP streams at `rtsp://relay:8554/s{0-5}`
- **Models**: Each pipeline uses different inference config from `/config/`
  - `s0`: ResNet Traffic (FP16) - 4 classes (Vehicle, Person, RoadSign, TwoWheeler)
  - `s1`: PeopleNet (INT8) - 3 classes (Person, Bag, Face)
  - `s2`: ResNet Detector (FP16) - General object detection
  - `s3`: City Segmentation (FP16) - 19 classes, semantic segmentation
  - `s4`: People Segmentation (Triton) - People semantic segmentation
  - `s5`: Default Test1 (FP16) - TrafficCamNet reference
- **GPU Sharing**: All 6 pipelines share single GPU via `--gpus all`
- **Volume Mounts**:
  - `-v $(pwd)/models:/models` - TensorRT engine cache (CRITICAL: use `$(pwd)` not `$PWD`)
  - `-v $(pwd)/config:/config` - Inference configs for different models

## Commits & Pull Requests
- Commit style: `scope: imperative summary` (e.g., `app: add queue element for smooth RTSP output`)
- PRs must include: change rationale, test steps (especially video quality), and doc updates when behavior changes
- Always test reconnection after modifications

## RTSP Reconnection Pattern (Python) - nvurisrcbin Method

**RECOMMENDED**: For Python DeepStream applications with RTSP sources, use **nvurisrcbin** with built-in reconnection:

1. **Use nvurisrcbin (not uridecodebin)**:
   ```python
   uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
   uri_decode_bin.set_property("uri", rtsp_uri)
   ```

2. **Configure reconnection properties**:
   ```python
   uri_decode_bin.set_property("rtsp-reconnect-interval", 10)  # Seconds between retries
   uri_decode_bin.set_property("init-rtsp-reconnect-interval", 5)  # Initial retry interval
   uri_decode_bin.set_property("rtsp-reconnect-attempts", -1)  # Infinite retries
   uri_decode_bin.set_property("select-rtp-protocol", 4)  # TCP-only (avoids UDP timeouts)
   ```

3. **Wrap in Bin with ghost pad** (NVIDIA pattern from deepstream-test3):
   ```python
   nbin = Gst.Bin.new("source-bin-00")
   Gst.Bin.add(nbin, uri_decode_bin)
   bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))

   # Connect pad-added to set ghost pad target when decoder pad appears
   uri_decode_bin.connect("pad-added", cb_newpad, nbin)
   ```

4. **cb_newpad sets ghost pad target**:
   ```python
   def cb_newpad(decodebin, decoder_src_pad, source_bin):
       if features.contains("memory:NVMM"):
           bin_ghost_pad = source_bin.get_static_pad("src")
           bin_ghost_pad.set_target(decoder_src_pad)
   ```

**Key Advantages**:
- No manual reconnection logic needed
- Pipeline stays PLAYING throughout
- nvurisrcbin handles all disconnects internally
- TCP-only avoids 5-second UDP timeout delays
- Works with both clean disconnects and abrupt closes

**Reference**: See `/root/d_final/app.py` for complete working implementation

## Notes for Agents
- Keep edits within the repo root; align with `STRUCTURE.md` and `STANDARDS.md`
- Avoid new frameworks; prefer surgical changes to `app.py`
- **DO NOT delete researched settings on a whim** - if a configuration was researched and implemented, validate thoroughly before removing
- **Research first, change second** - especially for encoder/timing/buffer settings
- GCP authentication: If running as root but gcloud/terraform are in prafulrana's home, check `gcloud auth list` FIRST before attempting application-default login. If an account is already authenticated, use `gcloud auth print-access-token` to get a token for Terraform. Export PATH: `export PATH="/usr/bin:/bin:/usr/local/bin:/home/prafulrana/google-cloud-sdk/bin"`
- Terraform on Ubuntu: Install from HashiCorp repo: `wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg && echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com noble main" | tee /etc/apt/sources.list.d/hashicorp.list && apt-get update && apt-get install -y terraform`

## Common Pitfalls & Checks
- **Missing queue elements cause choppy video** - Always include queue after encoder for frame buffering
- **Encoder optimization is critical** - Use `preset-id=0` (P1 highest performance) and `profile=2` (Main)
- **rtspclientsink needs latency** - Set `latency=200` (ms) for smooth RTP timing, otherwise warnings about "can't determine running time"
- **Don't confuse lag with choppiness** - Lag is latency (seconds), choppy is frame stuttering (buffering/timing issue)
- **TCP-only for both input/output** - `select-rtp-protocol=4` on nvurisrcbin, `protocols=0x00000004` on rtspclientsink
- **TensorRT engine caching** - Mount `/models` volume to avoid 30+ second rebuilds on every restart
- **Volume mount syntax** - CRITICAL: Use `$(pwd)` not `$PWD` in up.sh, otherwise mount fails silently
- **Ensure NVMM memory** - Check `cb_newpad` logs for "memory:NVMM" features, otherwise GPU decode failed
- **MediaMTX WebRTC timeouts** - Keep ICE/STUN/TURN timeout settings, removing them causes "reader too slow" issues

## Development Workflow: Fast Iteration

**Python file changes DO NOT require rebuild** - Python is interpreted, just restart container:
```bash
# After editing app.py or probe_*.py:
docker restart drishti-s2  # Changes take effect immediately (volume mounted)
```

**When rebuild IS required**:
- CUDA changes (`*.cu`, `build_cuda.sh`) → `./build.sh && ./up.sh`
- Dockerfile changes → `./build.sh && ./up.sh`
- Config changes (`config/*.txt`) → Just restart (volume mounted)

**Volume mounts for fast iteration** (already configured in `up.sh`):
```bash
-v "$(pwd)/app.py":/app/app.py                          # Main pipeline code
-v "$(pwd)/probe_default.py":/app/probe_default.py     # Probe modules
-v "$(pwd)/probe_yoloworld.py":/app/probe_yoloworld.py
-v "$(pwd)/probe_segmentation.py":/app/probe_segmentation.py
-v "$(pwd)/models":/models                              # TensorRT cache
-v "$(pwd)/config":/config                              # Inference configs
```

**Important**: Python bytecode caching can cause stale imports. If code changes don't take effect after restart:
```bash
docker exec drishti-s2 find /app -name "*.pyc" -delete
docker restart drishti-s2
```

## Pipeline Order & Probe Attachment: CRITICAL

**Correct pipeline order** (applies to ALL probe types):
```
pgie → nvvidconv (NV12→RGBA) → nvosd → rgba_caps → nvvidconv_postosd (RGBA→I420) → encoder
```

**Why this order matters**:
1. **pgie** outputs NV12 format
2. **nvvidconv** converts to RGBA (required for nvosd drawing in CPU mode)
3. **nvosd** draws bounding boxes and text on RGBA, AND finalizes segmentation metadata
4. **rgba_caps** ensures RGBA for probes
5. **nvvidconv_postosd** converts to I420 for encoder

**WRONG order** (causes no boxes to show): `pgie → nvosd → nvvidconv` (nvosd receives NV12, can't draw)

## Probe Attachment Patterns

Different probe types need different attachment points:

**Pattern 1: Custom tensor parsing (probe_yoloworld)**
- Probe must run BEFORE nvosd to add obj_meta for nvosd to draw
- Attach to: `nvvidconv.get_static_pad("src")` (right before nvosd input)
- Example: YOLOWorld custom tensor output parsing

**Pattern 2: Segmentation overlay (probe_segmentation)**
- Probe must run AFTER nvosd to access finalized segmentation metadata
- Attach to: `rgba_caps.get_static_pad("sink")` (after nvosd output)
- Example: Custom CUDA segmentation overlay

**Pattern 3: Default/no-op (probe_default)**
- Doesn't matter, can attach anywhere
- Typically attach after nvosd for consistency

**Implementation in app.py**:
```python
if args.probe == "probe_yoloworld":
    nvvidconv_srcpad = nvvidconv.get_static_pad("src")
    nvvidconv_srcpad.add_probe(Gst.PadProbeType.BUFFER, probe_module.osd_sink_pad_buffer_probe, 0)
else:
    rgba_sinkpad = rgba_caps.get_static_pad("sink")
    rgba_sinkpad.add_probe(Gst.PadProbeType.BUFFER, probe_module.osd_sink_pad_buffer_probe, 0)
```

## Custom CUDA Overlays: Segmentation

**When implementing custom CUDA overlays on segmentation masks**:

1. **Pipeline order**: Follow the standard order above (nvvidconv BEFORE nvosd)

2. **nvosd required**: Even if not using it for drawing, nvosd processes/finalizes segmentation metadata from nvinfer

3. **Data type conversion** - Segmentation masks from DeepStream are int32, CUDA kernels typically expect uint8:
   ```python
   masks = pyds.get_segmentation_masks(seg_meta)
   mask_array = np.array(masks, copy=True, order='C')
   mask_array = mask_array.astype(np.uint8)  # Convert int32 → uint8 for CUDA
   ```

4. **Mask values**: Binary segmentation (person/background) has values 0 (background) and 1 (foreground class)

5. **Common failure modes**:
   - All mask values are 0 → nvosd not in pipeline OR segmentation threshold too high
   - Sparse/offset overlay → Coordinate scaling mismatch between mask resolution and frame resolution
   - Random dots/garbage → Data type mismatch (int32 read as uint8) or stride issues

## YOLOWorld / Custom Tensor Parsing

**Issue**: DeepStream's `maintain-aspect-ratio=1` letterboxing may not match model's expectations

**Symptoms**:
- Wrong coordinates (boxes offset)
- Only certain classes detected (e.g., only bus, not person/car)
- Boxes in wrong positions

**Solution**: Use `maintain-aspect-ratio=0` for simple resize, then scale coordinates directly:

```python
# In config file:
maintain-aspect-ratio=0

# In probe:
net_w, net_h = 640, 640
scale_x = frame_width / net_w
scale_y = frame_height / net_h

# Direct scaling (NO letterbox unmapping):
x1 = x1 * scale_x
y1 = y1 * scale_y
x2 = x2 * scale_x
y2 = y2 * scale_y
```

**Don't use letterbox unmapping** with DeepStream's `maintain-aspect-ratio=1` unless you can verify the exact padding behavior matches your calculation.

## Performance Debugging: Don't Assume the Obvious

**Core Principle**: DeepStream has been proven to handle 64 concurrent 1080p streams with inference on this hardware. If a single 720p stream is failing, the issue is NOT raw performance - it's configuration or timing.

**When you see "reader too slow" or dropped frames**:

1. **Check buffer/timeout settings FIRST**:
   - `batched-push-timeout` (4000000 for live RTSP)
   - Queue properties (`max-size-buffers`, `max-size-time`)
   - Source buffering settings

2. **Check startup/state transitions**:
   - Is pipeline reaching PLAYING state?
   - Are elements prerolling correctly?
   - Is TensorRT engine cached or rebuilding?

3. **Check for actual bottlenecks**:
   - GPU utilization (`nvidia-smi`)
   - CPU usage per pipeline element (`GST_DEBUG=3`)
   - Network bandwidth/latency

**What NOT to do (unless you've proven the bottleneck)**:
- ❌ **Reduce resolution** - We handle 1080p @ 64 streams, 720p is trivial
- ❌ **Disable inference** - Defeats the purpose, not the bottleneck
- ❌ **Drop frames** - Wastes compute, doesn't fix timing issues
- ❌ **Lower bitrate** - Encoding is GPU-accelerated, not the bottleneck
- ❌ **Reduce concurrent streams** - This is a single-stream pipeline

**Common Root Causes** (in order of likelihood):
1. Buffer timeout too aggressive for pipeline warmup
2. TensorRT engine rebuilding on every run (missing volume mount)
3. Network issues (TCP retries, packet loss)
4. Wrong memory type (not using NVMM)
5. Actual compute bottleneck (check GPU utilization - unlikely)
