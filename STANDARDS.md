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

## Segmentation Config Requirements

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

## Common Mistakes to Avoid

1. ❌ Running docker commands manually instead of `./build.sh && ./up.sh`
2. ❌ Wrong pipeline order: `pgie → nvosd → nvvidconv` (nvosd needs RGBA)
3. ❌ Forgetting to rebuild after C++ changes (editing without `./build.sh`)
4. ❌ Using `network-type=100` instead of `network-type=2` for segmentation
5. ❌ Not removing old engine files when changing config
6. ❌ Removing nvosd from pipeline when using custom segmentation (metadata won't be finalized)
7. ❌ Copying segmentation data to CPU for processing (use GPU-only kernels)
8. ❌ Setting segmentation-threshold too high (causes sparse coverage)

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
