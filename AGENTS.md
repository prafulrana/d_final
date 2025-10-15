# Repository Guidelines

This repo hosts a **pure C++ DeepStream pipeline** for zero-copy GPU segmentation processing. The pipeline processes RTSP input with PeopleSegNet inference and outputs to RTSP via rtspclientsink. All segmentation overlay processing happens directly on GPU without CPU overhead.

## Project Structure & Modules
- `main.cpp` — Pure C++ GStreamer application: nvurisrcbin → nvstreammux → nvinfer → OSD → encoder → rtspclientsink
- `segmentation_probe_complete.cpp` — C++ pad probe for zero-copy GPU segmentation overlay
- `segmentation_overlay_direct.cu` — CUDA kernels for GPU-only overlay and int→float conversion
- `build_app.sh` — Builds C++ application with CUDA kernels
- `Dockerfile` — DeepStream 8.0 C++ environment with CUDA compilation
- `up.sh` — Pipeline orchestration script (runs s4 PeopleSemSegNet ONNX pipeline)
- `models/` — Persistent TensorRT engine cache (volume mounted)
- `config/` — nvinfer configurations (PeopleSemSegNet semantic segmentation)
- `relay/` — MediaMTX relay server configuration (GCP VM). Default zone: `asia-south1-c`
- `STANDARDS.md`, `STRUCTURE.md` — Build/test/debug documentation

## Build, Test, Run
- Build & Run: `./build.sh && ./up.sh` (builds `ds_python:latest`, runs s4 PeopleSemSegNet ONNX pipeline)
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

## Pure C++ Architecture
- **Input**: Single RTSP stream at `rtsp://relay:8554/in_s4`
- **Output**: Single RTSP stream at `rtsp://relay:8554/s4`
- **Model**: PeopleSemSegNet ONNX (peoplesemsegnet_shuffleseg.onnx) - Semantic segmentation (background, person)
- **Config**: network-type=2 with NvDsInferParseCustomPeopleSemSegNet parser
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

## Segmentation Probe: Zero-Copy GPU Implementation

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
