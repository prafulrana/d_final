# DeepStream Development Standards

## Testing Workflow

**ALWAYS use `./build.sh && ./up.sh` to test C++ changes.**

```bash
# Correct workflow for C++ changes:
# Make changes to main.cpp, probe, or CUDA files
./build.sh && ./up.sh        # Rebuild and restart pipeline

# Check logs:
docker logs -f drishti-s4

# View output:
http://RELAY_IP:8889/s4/
```

## Fast Iteration for Config Changes

Config files are **volume mounted**, no rebuild needed:

```bash
# After editing config/pgie_peoplesemseg_onnx.txt:
docker restart drishti-s4      # Changes take effect immediately
```

## C++ Changes Require Rebuild

C++ source files are **compiled into the image**:

```bash
# After editing main.cpp, segmentation_probe_complete.cpp, or *.cu:
./build.sh && ./up.sh          # Full rebuild required
```

## Pipeline Structure (CRITICAL ORDER)

**Correct order:**
```
nvurisrcbin → nvstreammux → nvinfer → nvvidconv → nvosd → rgba_caps →
  nvvidconv_postosd → capsfilter → nvv4l2h264enc → queue → h264parse → rtspclientsink
```

**Why this order matters:**
1. **nvvidconv BEFORE nvosd**: nvosd requires RGBA format to draw (CPU mode)
2. **queue after encoder**: Prevents choppy video

**WRONG order** (causes no boxes to show):
```
pgie → nvosd → nvvidconv  ❌  (nvosd receives NV12, can't draw)
```

## Segmentation Probe Pattern (C++)

**Current implementation:** Zero-copy GPU segmentation overlay

**Attachment point:** `rgba_caps.get_static_pad("sink")` (AFTER nvosd)

**Why:** Needs finalized segmentation metadata from nvosd

**Example (main.cpp):**
```cpp
// Attach probe AFTER nvosd to access finalized segmentation metadata
rgba_sinkpad = gst_element_get_static_pad(rgba_caps, "sink");
gst_pad_add_probe(rgba_sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
                  segmentation_probe_callback, NULL, NULL);
```

**Process flow:**
1. nvinfer outputs segmentation metadata
2. nvosd finalizes metadata (required!)
3. Probe accesses seg_meta → applies zero-copy GPU overlay

## YOLO Detection Config Requirements (s4)

```ini
[property]
network-type=0                        # Detector (not 2 for segmentation)
num-detected-classes=80               # COCO classes
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=/app/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet
maintain-aspect-ratio=0               # Disable for correct coordinate mapping

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
```

**Key settings:**
- `network-type=0`: Object detection (not 2 for segmentation)
- `parse-bbox-func-name`: Custom YOLO parser from DeepStream-Yolo library
- `maintain-aspect-ratio=0`: Required for correct bounding box coordinates
- Custom library: Built from https://github.com/marcoslucianops/DeepStream-Yolo

## Segmentation Config Requirements (s5)

```ini
[property]
network-type=2                        # Segmentation (not 100 for raw tensors)
num-detected-classes=2                # Background + person
segmentation-threshold=0.05           # Lower = more coverage, higher = less noise
parse-segmentation-func-name=NvDsInferParseCustomPeopleSemSegNet
custom-lib-path=/opt/nvidia/deepstream/deepstream-8.0/lib/libnvds_infercustomparser.so
```

**Key settings:**
- `network-type=2`: Enables segmentation metadata output (not raw tensors)
- `segmentation-threshold`: Balance between coverage and noise (0.05 works well)
- Custom parser: Converts INT64 model output to segmentation metadata

## Zero-Copy GPU Implementation

**C++ probe access pattern (segmentation_probe_complete.cpp):**

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

**CUDA kernels (segmentation_overlay_direct.cu):**
- `convert_int_to_float`: GPU-only int→float conversion (no CPU)
- `apply_segmentation_overlay_direct`: Green overlay with alpha blending

## nvosd Requirement

**Critical:** Even with custom CUDA overlay, nvosd MUST be in pipeline:

```cpp
// Pipeline: pgie → nvvidconv → nvosd → rgba_caps (probe) → ...
gst_bin_add_many(GST_BIN(pipeline), pgie, nvvidconv, nvosd, rgba_caps, ...);
gst_element_link_many(nvstreammux, pgie, nvvidconv, nvosd, rgba_caps, ...);
```

**Why:** nvosd processes/finalizes segmentation metadata from nvinfer

## TensorRT Engine Management

**Engine lifecycle:**
1. First run: Builds engine from ONNX (~5-10 min for 2048x2048)
2. Cached: Engine saved to `/models` volume (instant startup)
3. Shared: All 3 streams use same engine

### Engine Caching Workflow

**Standard workflow (recommended):**
```bash
# 1. Start pipelines (first run builds engine)
./up.sh

# 2. Monitor build progress
docker logs -f drishti-s0  # Watch for "Running main loop..."

# 3. Cache engine to host
./scripts/cache_engine.sh copy drishti-s0

# 4. Verify cached
./scripts/cache_engine.sh verify config/pgie_yolov8_coco.txt

# 5. Future runs use cache (instant startup)
./up.sh
```

### Engine Management Commands

```bash
# List all engines (containers + host)
./scripts/cache_engine.sh list

# Copy engine from container to host
./scripts/cache_engine.sh copy drishti-s0

# Verify engine for current config
./scripts/cache_engine.sh verify config/pgie_yolov8_coco.txt

# Clean all cached engines
./scripts/cache_engine.sh clean
```

### When to Rebuild Engine

**Must rebuild when:**
- Changed model (yolov8n → yolov8s)
- Changed resolution (1024 → 2048)
- Different GPU (engines are GPU-specific)
- TensorRT version changed
- ONNX file updated

**No rebuild needed when:**
- Config threshold changes (`pre-cluster-threshold`)
- NMS changes (`nms-iou-threshold`)
- Tracker config changes
- Encoder settings changes

### Multi-Stream First Run

**Problem**: 3 containers starting simultaneously all build engine (15-30 min)

**Solution**: Sequential first-run startup:
```bash
# Option 1: Start one stream first
docker run -d --name drishti-s0 --gpus all --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/libnvdsinfer_custom_impl_Yolo.so":/app/libnvdsinfer_custom_impl_Yolo.so \
  ds_python:latest \
  /app/deepstream_app \
  rtsp://34.14.140.30:8554/in_s0 \
  rtsp://34.14.140.30:8554/s0 \
  /config/pgie_yolov8_coco.txt

# Wait for build completion
docker logs -f drishti-s0

# Now start all 3 (use cached engine)
./up.sh

# Option 2: Pre-cache engine if available
./scripts/cache_engine.sh copy drishti-s0  # After any successful build
```

## Common Mistakes to Avoid

### General Pipeline Mistakes
1. ❌ Running docker commands manually instead of `./build.sh && ./up.sh`
2. ❌ Wrong pipeline order: `pgie → nvosd → nvvidconv` (nvosd needs RGBA)
3. ❌ Forgetting to rebuild after C++ changes (editing without `./build.sh`)
4. ❌ Not caching engine after first build (5-10 min startup every time)
5. ❌ Starting all 3 streams simultaneously on first run (engine builds 3x)

### YOLO Detection Mistakes
6. ❌ **Using standard Ultralytics ONNX export** - CRITICAL: Must use DeepStream-Yolo export script
7. ❌ Forgetting `weights_only=False` in export_yoloV8.py for PyTorch 2.6+
8. ❌ Missing custom parser library mount (`libnvdsinfer_custom_impl_Yolo.so`)
9. ❌ Using `network-type=2` instead of `network-type=0` for detection
10. ❌ Not setting `maintain-aspect-ratio=0` (causes coordinate misalignment)

### Segmentation Mistakes
11. ❌ Using `network-type=100` instead of `network-type=2` for segmentation
12. ❌ Removing nvosd from pipeline when using custom segmentation (metadata won't be finalized)
13. ❌ Copying segmentation data to CPU for processing (use GPU-only kernels)
14. ❌ Setting segmentation-threshold too high (causes sparse coverage)

## Git LFS

Model files are tracked with Git LFS:

```bash
# Check LFS files
git lfs ls-files

# Pull LFS files after clone
git lfs pull
```

**Tracked patterns:**
- `models/*.onnx`
- `models/*.etlt`
- `models/*.tflite`
- `models/*.zip`

**NOT tracked** (in `.gitignore`):
- `models/*.engine` (TensorRT cache, GPU-specific)

## Build System

**For C++ changes (main.cpp, probe, CUDA):** Full rebuild required
```bash
./build.sh && ./up.sh
```

**For config changes:** Just restart (volume mounted)
```bash
docker restart drishti-s4
```
