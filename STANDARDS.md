# DeepStream Development Standards

## Testing Workflow

**ALWAYS use `./up.sh` to test changes. NEVER run docker commands manually.**

```bash
# Correct workflow:
./down.sh      # Stop pipelines
# Make changes to code/config
./up.sh        # Rebuild and restart all pipelines

# Check logs:
docker logs drishti-s0
docker logs drishti-s1
docker logs drishti-s2

# View outputs:
http://RELAY_IP:8889/s0/
http://RELAY_IP:8889/s1/
http://RELAY_IP:8889/s2/
```

## Fast Iteration for Python Changes

Python files are **volume mounted**, no rebuild needed:

```bash
# After editing app.py or probe_*.py:
docker restart drishti-s2      # Changes take effect immediately

# If changes don't apply (Python bytecode cache):
docker exec drishti-s2 find /app -name "*.pyc" -delete
docker restart drishti-s2
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

## Probe Attachment Patterns

Different probe types attach at different points in the pipeline:

### Pattern 1: Custom Tensor Parsing (probe_yoloworld)

**When to use:** Model outputs raw tensors that need manual parsing before nvosd can draw

**Attachment point:** `nvvidconv.get_static_pad("src")` (BEFORE nvosd)

**Why:** Must add `obj_meta` before nvosd processes the frame for drawing

**Example:**
```python
if args.probe == "probe_yoloworld":
    nvvidconv_srcpad = nvvidconv.get_static_pad("src")
    nvvidconv_srcpad.add_probe(Gst.PadProbeType.BUFFER, probe_module.osd_sink_pad_buffer_probe, 0)
```

**Process flow:**
1. nvinfer outputs tensor metadata
2. Probe parses tensor → creates obj_meta
3. nvosd draws boxes based on obj_meta

### Pattern 2: Segmentation Overlay (probe_segmentation)

**When to use:** Custom visualization of segmentation masks (bypassing nvosd)

**Attachment point:** `rgba_caps.get_static_pad("sink")` (AFTER nvosd)

**Why:** Needs finalized segmentation metadata from nvosd

**Example:**
```python
else:  # probe_segmentation or probe_default
    rgba_sinkpad = rgba_caps.get_static_pad("sink")
    rgba_sinkpad.add_probe(Gst.PadProbeType.BUFFER, probe_module.osd_sink_pad_buffer_probe, 0)
```

**Process flow:**
1. nvinfer outputs segmentation metadata
2. nvosd finalizes metadata (required!)
3. Probe accesses seg_meta → applies custom CUDA overlay

### Pattern 3: Default/No-op (probe_default)

**When to use:** Standard detection with nvosd drawing

**Attachment point:** `rgba_caps.get_static_pad("sink")` (AFTER nvosd)

**Why:** Doesn't matter (probe does nothing)

**Example:**
```python
def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Default probe - no custom processing"""
    return Gst.PadProbeReturn.OK
```

## Custom Tensor Parsing

### Config Requirements (nvinfer)

```ini
[property]
network-type=100          # REQUIRED: Disables built-in parsing
output-tensor-meta=1      # REQUIRED: Enables tensor metadata
network-mode=2            # 0=FP32, 1=INT8, 2=FP16
maintain-aspect-ratio=0   # CRITICAL for YOLOWorld coordinate mapping
```

### Tensor Access (Python)

**WRONG ❌**
```python
ptr = pyds.get_ptr(layer.buffer)
```

**CORRECT ✅**
```python
# Get base pointer array, then offset by layer index
ptr_array = pyds.get_ptr(tensor_meta.out_buf_ptrs_host)
ptr = ctypes.cast(ptr_array, ctypes.POINTER(ctypes.c_void_p))[i]

# Create numpy array
flat_data = np.ctypeslib.as_array(
    ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
    shape=(N * 6,)
)
detections = flat_data.reshape(N, 6)
```

### YOLOWorld Coordinate Mapping

**Issue:** DeepStream's `maintain-aspect-ratio=1` letterboxing doesn't match manual calculation

**Solution:** Use `maintain-aspect-ratio=0` for simple resize:

```ini
# In config file:
maintain-aspect-ratio=0
```

```python
# In probe (direct scaling, NO letterbox unmapping):
net_w, net_h = 640, 640
scale_x = frame_width / net_w
scale_y = frame_height / net_h

x1 = x1 * scale_x  # NOT: (x1 - padX) / scale
y1 = y1 * scale_y
```

**Why:** DeepStream's letterboxing behavior is unpredictable with `maintain-aspect-ratio=1`

## Segmentation Overlay

### Data Type Conversion

Segmentation masks from DeepStream are **int32**, CUDA kernels expect **uint8**:

```python
masks = pyds.get_segmentation_masks(seg_meta)
mask_array = np.array(masks, copy=True, order='C')
mask_array = mask_array.astype(np.uint8)  # Convert int32 → uint8
```

### nvosd Requirement

**Critical:** Even if using custom CUDA overlay, nvosd MUST be in pipeline:

```python
# Pipeline: pgie → nvvidconv → nvosd → rgba_caps (probe) → ...
g_pipeline.add(pgie)
g_pipeline.add(nvvidconv)
g_pipeline.add(nvosd)  # REQUIRED - finalizes segmentation metadata
```

**Why:** nvosd processes/finalizes segmentation metadata from nvinfer

## Model Input Format

- **NCHW**: `infer-dims=3;256;256` (channels, height, width)
- **NHWC**: `infer-dims=256;256;3` (height, width, channels)
- Add `network-input-order=1` for NHWC if needed

## Common Mistakes to Avoid

1. ❌ Running random docker commands instead of `./up.sh`
2. ❌ Wrong pipeline order: `pgie → nvosd → nvvidconv` (nvosd needs RGBA)
3. ❌ Attaching probe_yoloworld AFTER nvosd (obj_meta won't be drawn)
4. ❌ Using `maintain-aspect-ratio=1` with manual coordinate mapping
5. ❌ Using `layer.buffer` instead of `out_buf_ptrs_host` for tensor access
6. ❌ Forgetting `network-type=100` for custom parsing
7. ❌ Not removing old engine files when changing config
8. ❌ Removing nvosd from pipeline when using custom segmentation (metadata won't be finalized)
9. ❌ Not converting int32 mask to uint8 before passing to CUDA

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

**For Python changes only:** Just restart container
```bash
docker restart drishti-s2
```

**For CUDA/Dockerfile changes:** Full rebuild required
```bash
./build.sh && ./up.sh
```

**For config changes:** Just restart (volume mounted)
```bash
docker restart drishti-s2
```
