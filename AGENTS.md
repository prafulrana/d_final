# Repository Guidelines

This repo hosts **DeepStream 8.0 pipelines** for object detection and semantic segmentation. The pipelines process RTSP input with NVIDIA TensorRT inference and output to RTSP via rtspclientsink. Supports both YOLO detection (s4) and PeopleSemSegNet segmentation (s5).

## Project Structure & Modules
- `main.cpp` — Pure C++ GStreamer application: nvurisrcbin → nvstreammux → nvinfer → OSD → encoder → rtspclientsink
- `segmentation_probe_complete.cpp` — C++ pad probe for zero-copy GPU segmentation overlay (s5 only)
- `segmentation_overlay_direct.cu` — CUDA kernels for GPU-only overlay and int→float conversion (s5 only)
- `libnvdsinfer_custom_impl_Yolo.so` — Custom YOLO parser library (s4 only)
- `build_app.sh` — Builds C++ application with CUDA kernels
- `Dockerfile` — DeepStream 8.0 C++ environment with CUDA compilation
- `up.sh` — Pipeline orchestration script (runs s4 YOLOv8 COCO detection)
- `models/` — Persistent TensorRT engine cache + ONNX models (volume mounted)
- `config/` — nvinfer configurations (YOLOv8 detection, PeopleSemSegNet segmentation)
- `relay/` — MediaMTX relay server configuration (GCP VM). Default zone: `asia-south1-c`
- `STANDARDS.md`, `STRUCTURE.md` — Build/test/debug documentation
- `DeepStream-Yolo/` — Community YOLO support repo (custom parser + export tools)

## Build, Test, Run
- Build & Run: `./build.sh && ./up.sh` (builds `ds_python:latest`, runs s4 YOLOv8 COCO detection)
- View logs: `docker logs -f drishti-s4`
- Monitor: `docker ps --filter name=drishti-s4`
- Publish test stream: Use Larix app or `gst-launch-1.0` to `rtsp://server:8554/in_s4`
- View output: `http://server:8889/s4/` (WebRTC)
- Relay deploy: `cd relay && terraform init && terraform apply -var project_id=<GCP_PROJECT>`

## Coding Style & Conventions
- Pure C++ with GStreamer C API
- 4-space indentation, max 100 character lines
- Follow NVIDIA DeepStream C/C++ sample patterns
- Use `g_print()` for info, `g_printerr()` for errors
- Keep functions focused with early returns
- CUDA kernels follow standard NVIDIA patterns (16x16 thread blocks)

## Testing Guidelines
- **Initial connection test**: Publish to `in_s4`, verify output at `s4`, check logs for "Pipeline is PLAYING"
- **Reconnection test**: Stop publisher, wait ~10s for reconnect logs, restart publisher, verify auto-recovery
- **Video quality test**: Compare smoothness of `in_s4` vs input - output should match input quality
- **Performance test**: Verify startup time <10s with cached TensorRT engine, CPU usage <5%
- **Segmentation test**: Verify green overlay appears on detected people (excellent mask coverage)
- Include logs and minimal repro for any pipeline or CUDA kernel changes

## Pipeline Architectures

### s4: YOLOv8 COCO Detection (Active)
- **Input**: Single RTSP stream at `rtsp://relay:8554/in_s4`
- **Output**: Single RTSP stream at `rtsp://relay:8554/s4`
- **Model**: YOLOv8n ONNX (yolov8n.onnx) - Object detection (80 COCO classes)
- **Config**: `/config/pgie_yolov8_coco.txt` with `network-type=0` (detector)
- **Custom Parser**: `libnvdsinfer_custom_impl_Yolo.so` (DeepStream-Yolo community parser)
- **Processing**: Standard nvosd bounding box drawing
- **Performance**: <5% CPU usage, TensorRT GPU inference
- **Volume Mounts**:
  - `-v $(pwd)/models:/models` - TensorRT engine cache + ONNX models (CRITICAL: use `$(pwd)` not `$PWD`)
  - `-v $(pwd)/config:/config` - Inference config (pgie_yolov8_coco.txt)
  - `-v $(pwd)/libnvdsinfer_custom_impl_Yolo.so:/app/libnvdsinfer_custom_impl_Yolo.so` - Custom YOLO parser

### s5: PeopleSemSegNet Segmentation
- **Input**: Single RTSP stream at `rtsp://relay:8554/in_s5`
- **Output**: Single RTSP stream at `rtsp://relay:8554/s5`
- **Model**: PeopleSemSegNet ONNX (peoplesemsegnet_shuffleseg.onnx) - Semantic segmentation (background, person)
- **Config**: `/config/pgie_peoplesemseg_onnx.txt` with `network-type=2` (segmentation)
- **Processing**: Zero-copy GPU segmentation overlay with custom CUDA kernels
- **Performance**: <5% CPU usage, all processing on GPU
- **Volume Mounts**:
  - `-v $(pwd)/models:/models` - TensorRT engine cache (CRITICAL: use `$(pwd)` not `$PWD`)
  - `-v $(pwd)/config:/config` - Inference config (pgie_peoplesemseg_onnx.txt)

## Commits & Pull Requests
- Commit style: `scope: imperative summary` (e.g., `app: add queue element for smooth RTSP output`)
- PRs must include: change rationale, test steps (especially video quality), and doc updates when behavior changes
- Always test reconnection after modifications

## RTSP Reconnection Pattern (C++) - nvurisrcbin Method

**RECOMMENDED**: For C++ DeepStream applications with RTSP sources, use **nvurisrcbin** with built-in reconnection:

1. **Create nvurisrcbin**:
   ```cpp
   source = gst_element_factory_make("nvurisrcbin", "source");
   g_object_set(G_OBJECT(source), "uri", rtsp_in, NULL);
   ```

2. **Configure reconnection properties**:
   ```cpp
   g_object_set(G_OBJECT(source), "rtsp-reconnect-interval", 10, NULL);  // Seconds between retries
   g_object_set(G_OBJECT(source), "init-rtsp-reconnect-interval", 5, NULL);  // Initial retry
   g_object_set(G_OBJECT(source), "rtsp-reconnect-attempts", -1, NULL);  // Infinite retries
   g_object_set(G_OBJECT(source), "select-rtp-protocol", 4, NULL);  // TCP-only
   ```

3. **Handle dynamic pads with pad-added callback**:
   ```cpp
   typedef struct {
       GstElement *nvstreammux;
       int stream_id;
   } PadData;

   g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), pad_data);
   ```

4. **on_pad_added links to nvstreammux**:
   ```cpp
   static void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
       PadData *pad_data = (PadData *)data;
       gchar pad_name[16];
       snprintf(pad_name, 15, "sink_%u", pad_data->stream_id);
       GstPad *sinkpad = gst_element_request_pad_simple(pad_data->nvstreammux, pad_name);
       gst_pad_link(pad, sinkpad);
       gst_object_unref(sinkpad);
   }
   ```

**Key Advantages**:
- No manual reconnection logic needed
- Pipeline stays PLAYING throughout
- nvurisrcbin handles all disconnects internally
- TCP-only avoids 5-second UDP timeout delays
- Works with both clean disconnects and abrupt closes

**Reference**: See `/root/d_final/main.cpp` for complete working implementation

## Notes for Agents
- Keep edits within the repo root; align with `STRUCTURE.md` and `STANDARDS.md`
- Avoid new frameworks; prefer surgical changes to C++ source files
- **DO NOT delete researched settings on a whim** - if a configuration was researched and implemented, validate thoroughly before removing
- **Research first, change second** - especially for encoder/timing/buffer settings
- **C++ changes require rebuild** - Run `./build.sh && ./up.sh` after modifying main.cpp, probe, or CUDA files
- **Config changes don't require rebuild** - Just restart container after modifying pgie_peoplesegnet.txt
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

**C++ file changes REQUIRE rebuild** - Compiled application must be rebuilt:
```bash
# After editing main.cpp, segmentation_probe_complete.cpp, or *.cu:
./build.sh && ./up.sh  # Rebuilds C++ executable and restarts container
```

**When rebuild IS NOT required**:
- Config changes (`config/pgie_peoplesemseg_onnx.txt`) → Just restart (volume mounted):
```bash
docker restart drishti-s4
```

**Volume mounts** (configured in `up.sh`):
```bash
-v "$(pwd)/models":/models   # TensorRT cache
-v "$(pwd)/config":/config   # Inference config (pgie_peoplesemseg_onnx.txt, can modify without rebuild)
```

## Pipeline Order & Probe Attachment: CRITICAL

**Correct pipeline order**:
```
pgie → nvvidconv (NV12→RGBA) → nvosd → rgba_caps → nvvidconv_postosd (RGBA→I420) → encoder
```

**Why this order matters**:
1. **pgie** outputs NV12 format
2. **nvvidconv** converts to RGBA (required for nvosd drawing in CPU mode)
3. **nvosd** draws bounding boxes and text on RGBA, AND finalizes segmentation metadata
4. **rgba_caps** ensures RGBA for probe
5. **nvvidconv_postosd** converts to I420 for encoder

**WRONG order** (causes no boxes to show): `pgie → nvosd → nvvidconv` (nvosd receives NV12, can't draw)

## YOLO Detection Configuration (s4)

**DeepStream-YOLO Export Process** (CRITICAL for proper bounding boxes):

YOLOv8 models require **DeepStream-specific ONNX export** for correct output format:

1. **Use DeepStream-Yolo export script** (not standard Ultralytics export):
   ```bash
   # Standard export WILL NOT WORK with DeepStream parser
   # yolo export model=yolov8n.pt format=onnx  ❌

   # Use DeepStream-Yolo custom export script:
   python3 DeepStream-Yolo/utils/export_yoloV8.py -w yolov8n.pt --dynamic -s 640
   ```

2. **Why custom export is required**:
   - Standard Ultralytics ONNX includes post-processing incompatible with DeepStream
   - DeepStream-Yolo adds `DeepStreamOutput` layer that formats: `[boxes, scores, labels]`
   - Without this layer, bounding boxes will be completely misaligned

3. **PyTorch 2.6+ compatibility fix**:
   - Line 38 of `export_yoloV8.py` needs `weights_only=False`:
   ```python
   ckpt = torch.load(weights, map_location='cpu', weights_only=False)
   ```

4. **YOLO config requirements** (pgie_yolov8_coco.txt):
   ```ini
   [property]
   network-type=0                                      # Detector (not 2 for segmentation)
   num-detected-classes=80                             # COCO classes
   parse-bbox-func-name=NvDsInferParseYolo
   custom-lib-path=/app/libnvdsinfer_custom_impl_Yolo.so
   engine-create-func-name=NvDsInferYoloCudaEngineGet
   maintain-aspect-ratio=0                             # Disable for correct coordinate mapping

   [class-attrs-all]
   nms-iou-threshold=0.45
   pre-cluster-threshold=0.25
   topk=300
   ```

5. **Custom parser library**:
   - Source: https://github.com/marcoslucianops/DeepStream-Yolo
   - Build inside DS8 container: `export CUDA_VER=12.8 && make`
   - Output: `libnvdsinfer_custom_impl_Yolo.so` (~1.2MB)
   - Mount to container: `-v $(pwd)/libnvdsinfer_custom_impl_Yolo.so:/app/libnvdsinfer_custom_impl_Yolo.so`

6. **Common YOLO failure modes**:
   - Misaligned bounding boxes → Used standard Ultralytics export instead of DeepStream-Yolo export
   - No detections → Wrong parser function or missing custom library
   - Crashes on load → CUDA version mismatch in parser library build
   - Engine path issues → Engine saves to `/app/model_b1_gpu0_fp16.engine` instead of configured path

## Segmentation Probe: Zero-Copy GPU Implementation (s5)

**C++ probe attachment** (in main.cpp):
```cpp
// Attach probe AFTER nvosd to access finalized segmentation metadata
rgba_sinkpad = gst_element_get_static_pad(rgba_caps, "sink");
gst_pad_add_probe(rgba_sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
                  segmentation_probe_callback, NULL, NULL);
```

**Key implementation details**:

1. **Pipeline order**: nvvidconv BEFORE nvosd (nvosd requires RGBA)

2. **nvosd required**: Even with custom CUDA overlay, nvosd processes/finalizes segmentation metadata from nvinfer

3. **Zero-copy GPU processing** (segmentation_probe_complete.cpp):
   ```cpp
   // Get GPU frame pointer directly from NvBufSurface
   NvBufSurface *surf = (NvBufSurface *)map_info.data;
   void *frame_gpu_ptr = surf->surfaceList[frame_meta->batch_id].dataPtr;

   // Get segmentation metadata (int class_map on CPU)
   NvDsInferSegmentationMeta *seg_meta = (NvDsInferSegmentationMeta *)user_meta->user_meta_data;

   // GPU-only int→float conversion (ZERO CPU overhead)
   cudaMalloc((void**)&class_map_gpu, seg_size * sizeof(int));
   cudaMemcpy(class_map_gpu, seg_meta->class_map, seg_size * sizeof(int), cudaMemcpyHostToDevice);

   cudaMalloc((void**)&seg_float_gpu, seg_size * sizeof(float));
   convert_classmap_gpu(class_map_gpu, seg_float_gpu, seg_size);  // GPU kernel

   // Launch overlay kernel (pure GPU)
   launch_segmentation_overlay_direct(frame_gpu_ptr, seg_float_gpu, ...);
   ```

4. **CUDA kernels** (segmentation_overlay_direct.cu):
   - `convert_int_to_float`: GPU-only int→float conversion
   - `apply_segmentation_overlay_direct`: Green overlay with alpha blending on person pixels

5. **Segmentation config**: Set in config file (pgie_peoplesemseg_onnx.txt):
   ```ini
   network-type=2                                      # Segmentation (not 100 for raw tensors)
   segmentation-threshold=0.05                         # Lower = more coverage, higher = less noise
   parse-segmentation-func-name=NvDsInferParseCustomPeopleSemSegNet
   custom-lib-path=/opt/nvidia/deepstream/deepstream-8.0/lib/libnvds_infercustomparser.so
   ```

6. **Common failure modes**:
   - All mask values are 0 → Wrong network-type (use 2, not 100) OR threshold too high
   - Model outputs zeros → Wrong preprocessing or network-type=100 without proper tensor parsing
   - Sparse/offset overlay → Coordinate scaling mismatch or threshold too high
   - High CPU usage → Not using GPU-only conversion, copying data to CPU

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

## NvDCF Tracker Configuration

**CRITICAL**: NvDCF tracker parameter names are EXACT - wrong names cause silent failures or crashes.

### Valid StateEstimator Parameters

**For `stateEstimatorType: 1` (SIMPLE Kalman Filter)**:

```yaml
StateEstimator:
  stateEstimatorType: 1    # SIMPLE state estimator
  processNoiseVar4Loc: 1.5     # Process noise for bbox center
  processNoiseVar4Size: 1.3    # Process noise for bbox size
  processNoiseVar4Vel: 0.03    # Process noise for velocity
  measurementNoiseVar4Detector: 3.0
  measurementNoiseVar4Tracker: 8.0
```

**Common mistakes**:
- ❌ `noiseWeightVar4Loc` → Causes "Unknown param" warnings
- ❌ `noiseWeightVar4Acc` → Does NOT exist, causes memory corruption + crash
- ✅ `processNoiseVar4Loc` → Correct parameter name

### Smooth Bounding Boxes

**Problem**: Boxes jitter/resize by 1-2% every frame

**Solution**: Increase process noise variance (trust past state more):

```yaml
StateEstimator:
  processNoiseVar4Loc: 3.0    # Higher = smoother position (default: 1.5)
  processNoiseVar4Size: 3.0   # Higher = smoother size (default: 1.3)
  processNoiseVar4Vel: 0.05   # Slightly higher for smoother velocity

VisualTracker:
  filterLr: 0.02    # Lower learning rate = smoother visual tracking (default: 0.075)
```

### Persistent Tracking (Reduce Flickering)

**Problem**: Objects appear/disappear every second

**Solution**: Lower thresholds and increase shadow tracking:

```yaml
BaseConfig:
  minDetectorConfidence: 0.15    # Lower to accept more detections

TargetManagement:
  minTrackerConfidence: 0.2      # Lower for shadow tracking
  probationAge: 1                # Confirm tracks faster
  maxShadowTrackingAge: 90       # Keep tracks alive 90 frames (3s @ 30fps)
```

**Also check detection config** (`pgie_yolov8_coco.txt`):
```ini
pre-cluster-threshold=0.15    # Lower = more detections
```

### Tracker Config Debugging

**Check for invalid parameters**:
```bash
docker logs drishti-s0 2>&1 | grep "WARNING.*Unknown param"
```

**Common crash signature**:
```
!! [WARNING][SimpleEstimatorParams] Unknown param found: noiseWeightVar4Acc
corrupted size vs. prev_size
Aborted (core dumped)
```

## YOLOv8 Model Export for DeepStream

**CRITICAL**: Standard Ultralytics export does NOT work with DeepStream parsers.

### Correct Export Method

**Use DeepStream-Yolo community export script**:

```bash
# Automated script (recommended)
./scripts/export_yolov8.sh yolov8n 2048

# Manual export
docker run --rm \
    -v /root/d_final/DeepStream-Yolo:/deepstream-yolo \
    -v /root/d_final/models:/models \
    ultralytics/ultralytics:latest bash -c "
        pip install -q onnx onnxsim && \
        cd /models && \
        python3 /deepstream-yolo/utils/export_yoloV8.py -w yolov8n.pt --dynamic -s 2048
    "
```

**Why this is required**:
- Standard export: `[boxes, scores, classes]` as separate tensors
- DeepStream format: Single `DeepStreamOutput` layer with specific structure
- Custom parser (`libnvdsinfer_custom_impl_Yolo.so`) expects DeepStream format

### PyTorch 2.6+ Compatibility

**Error**: `_pickle.UnpicklingError: Weights only load failed`

**Fix**: Modify `DeepStream-Yolo/utils/export_yoloV8.py` line 38:
```python
ckpt = torch.load(weights, map_location='cpu', weights_only=False)  # Add weights_only=False
```

### Export Workflow

1. **Download model** (if not exists):
   ```bash
   ./scripts/download_model.sh yolov8 yolov8n
   ```

2. **Export to ONNX**:
   ```bash
   ./scripts/export_yolov8.sh yolov8n 2048
   # Creates: models/yolov8n_2048.onnx
   ```

3. **Update config**:
   ```ini
   onnx-file=/models/yolov8n_2048.onnx
   model-engine-file=/models/yolov8n_2048_b1_gpu0_fp16.engine
   ```

4. **First run builds TensorRT engine** (~5-10 min for 2048x2048):
   ```bash
   ./up.sh
   docker logs -f drishti-s0  # Watch for "Building the TensorRT Engine"
   ```

5. **Subsequent runs use cached engine** (instant startup)

### Resolution Selection

| Resolution | Inference Speed | Detection Quality | Use Case |
|-----------|----------------|-------------------|----------|
| 640x640 | ~120 FPS | Good for large objects | High FPS, close-range |
| 1024x1024 | ~80 FPS | Balanced | General purpose |
| 1280x1280 | ~60 FPS | Better small objects | Medium distance |
| 2048x2048 | ~30 FPS | Best small objects | Long distance, detail |

**Hardware context**: With NVIDIA 5070Ti/5080, even 3x concurrent 2048x2048 streams run at 30+ FPS.

## Common Pipeline Issues

### Issue: Containers Crash Immediately

**Symptom**: `docker ps -a` shows `Exited (134)` status

**Debug**:
1. Remove `--rm` flag from `up.sh` to preserve crashed containers
2. Check logs: `docker logs drishti-s0`
3. Look for: "Unknown param", "corrupted size", "Aborted"

**Common causes**:
- Invalid tracker config parameters (see NvDCF section above)
- Missing custom parser library
- Wrong ONNX format (use DeepStream-Yolo export)

### Issue: Jerky/Non-smooth Frames

**Symptom**: Video stutters despite good FPS

**Root cause**: `batched-push-timeout` too high (frames batching)

**Fix**: Lower timeout in `main.cpp`:
```cpp
g_object_set(G_OBJECT(nvstreammux), "batched-push-timeout", 40000, NULL);  // 40ms
```

### Issue: Motion Artifacts/Pixelation

**Symptom**: Blocky squares during fast motion

**Root cause**: Bitrate too low for complexity

**Fix**: Increase bitrate or use slower preset:
```cpp
g_object_set(G_OBJECT(encoder), "bitrate", 8000000, NULL);     // 8Mbps
g_object_set(G_OBJECT(encoder), "preset-level", 3, NULL);      // Slow = quality
```

### Issue: Bounding Boxes Misaligned

**Symptom**: Detection boxes don't match objects (see screenshot examples)

**Root cause**: Wrong ONNX export format

**Fix**: Re-export with DeepStream-Yolo script (NOT standard Ultralytics):
```bash
./scripts/export_yolov8.sh yolov8n 2048
```


### Issue: Slow Pipeline Startup (5-10 min)

**Symptom**: Pipeline takes 5-10 minutes to start on every run

**Root cause**: TensorRT engine not cached, rebuilding every time

**Debug**:
```bash
# Check if engine exists on host
./scripts/cache_engine.sh verify config/pgie_yolov8_coco.txt

# List engines
./scripts/cache_engine.sh list
```

**Fix**: Cache engine after first successful build:
```bash
# 1. Wait for first build to complete
docker logs -f drishti-s0  # Watch for "Running main loop..."

# 2. Copy engine to host
./scripts/cache_engine.sh copy drishti-s0

# 3. Verify cached
ls -lh models/*.engine

# 4. Future runs will use cache (instant startup)
./up.sh
```

**Common mistakes:**
- `/models` not volume-mounted → Engine builds in container, lost on restart
- Engine path in config doesn't match actual location
- GPU changed → Engine is GPU-specific, must rebuild

### Issue: Engine Build Failed

**Symptom**: Pipeline exits during "Building the TensorRT Engine"

**Debug**:
```bash
# Check full logs
docker logs drishti-s0 2>&1 | grep -A 20 "Building the TensorRT Engine"
```

**Common causes:**
1. **Out of GPU memory**: Reduce model size or resolution
2. **Wrong ONNX format**: Re-export with DeepStream-Yolo script
3. **CUDA version mismatch**: Check CUDA version in logs
4. **Corrupted ONNX file**: Re-export model

**Fix**:
```bash
# Clean old engines
./scripts/cache_engine.sh clean

# Re-export ONNX
./scripts/export_yolov8.sh yolov8n 2048

# Retry
./up.sh
```

### Issue: Multiple Containers Building Same Engine

**Symptom**: All 3 streams (s0, s1, s2) building engine separately, taking 15-30 min total

**Root cause**: Containers started simultaneously before engine cached

**Fix**: Sequential startup for first run:
```bash
# 1. Start first stream only
docker run -d --name drishti-s0 --gpus all --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/libnvdsinfer_custom_impl_Yolo.so":/app/libnvdsinfer_custom_impl_Yolo.so \
  ds_python:latest \
  /app/deepstream_app \
  rtsp://34.14.140.30:8554/in_s0 \
  rtsp://34.14.140.30:8554/s0 \
  /config/pgie_yolov8_coco.txt

# 2. Wait for engine build
docker logs -f drishti-s0  # Wait for "Running main loop..."

# 3. Now start all 3 streams (will use cached engine)
./up.sh
```

**Or use cache_engine.sh:**
```bash
# After first successful run
./scripts/cache_engine.sh copy drishti-s0

# Future ./up.sh runs will use cached engine for all 3 streams
```
