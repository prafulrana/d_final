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
```

## DeepStream Tensor Metadata Access

**Reference**: `/root/deepstream-8.0/sources/apps/sample_apps/deepstream-infer-tensor-meta-test/`

### Correct Config (nvinfer)
```ini
[property]
network-type=100          # REQUIRED: Disables built-in parsing
output-tensor-meta=1      # REQUIRED: Enables tensor metadata
network-mode=0            # 0=FP32, 1=INT8, 2=FP16
```

### Correct Tensor Access (Python)
```python
# WRONG ❌
ptr = pyds.get_ptr(layer.buffer)

# CORRECT ✅ (from C++ example line 230)
ptr = tensor_meta.out_buf_ptrs_host[i]

# Create numpy array
heatmaps = np.ctypeslib.as_array(
    (np.ctypeslib.ctypes.c_float * total_elements).from_address(ptr),
    shape=(num_keypoints, h, w)
)
```

## Model Input Format

- **NCHW**: `infer-dims=3;256;256` (channels, height, width)
- **NHWC**: `infer-dims=256;256;3` (height, width, channels)
- Add `network-input-order=1` for NHWC if needed

## Pipeline Structure

```
nvurisrcbin → nvstreammux → nvinfer → nvvideoconvert → nvdsosd →
  nvvideoconvert → capsfilter → nvv4l2h264enc → queue → h264parse → rtspclientsink
```

## Common Mistakes to Avoid

1. ❌ Running random docker commands instead of `./up.sh`
2. ❌ Using `layer.buffer` instead of `out_buf_ptrs_host`
3. ❌ Forgetting `network-type=100` for custom parsing
4. ❌ Not removing old engine files when changing config
5. ❌ Making changes without testing with `./up.sh`
