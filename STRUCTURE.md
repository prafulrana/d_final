# Project Structure

```
d_final/
├── main.cpp                              # Pure C++ GStreamer application
├── segmentation_probe_complete.cpp       # C++ pad probe for zero-copy GPU overlay (s5 only)
├── segmentation_overlay_direct.cu        # CUDA kernels (overlay, int→float conversion) (s5 only)
├── libnvdsinfer_custom_impl_Yolo.so      # Custom YOLO parser library (s4 only)
├── build_app.sh                          # Builds C++ application with CUDA
├── Dockerfile                            # DeepStream 8.0 C++ + CUDA build environment
├── build.sh                              # Build Docker image
├── up.sh                                 # Start s4 YOLOv8 COCO detection pipeline
├── config/
│   ├── pgie_yolov8_coco.txt             # YOLOv8 COCO detection config (s4)
│   └── pgie_peoplesemseg_onnx.txt       # PeopleSemSegNet ONNX config (s5)
├── models/                               # Model files (Git LFS)
│   ├── yolov8n.onnx                     # YOLOv8 nano detection (13MB, DeepStream format)
│   ├── coco_labels.txt                  # COCO class labels (80 classes)
│   ├── peoplesemsegnet_vdeployable_shuffleseg_unet_onnx_v1.0.1/
│   │   ├── peoplesemsegnet_shuffleseg.onnx  # Semantic segmentation model (3.8MB)
│   │   ├── labels.txt
│   │   └── peoplesemsegnet_shuffleseg_int8.txt
│   └── *.engine                         # TensorRT cache (not in git)
├── DeepStream-Yolo/                      # Community YOLO support (cloned)
│   ├── nvdsinfer_custom_impl_Yolo/      # Custom parser source
│   └── utils/export_yoloV8.py           # DeepStream-specific ONNX export
├── relay/                                # MediaMTX relay infrastructure (GCP)
│   ├── main.tf                          # Terraform config
│   ├── variables.tf                     # Variables
│   ├── scripts/startup.sh               # VM startup script
│   └── README.md                        # Deployment guide
├── AGENTS.md                             # AI agent guidelines
├── STANDARDS.md                          # Development standards
├── STRUCTURE.md                          # This file
└── plan.md                               # YOLO implementation milestone plan
```

## Architecture Overview

### DeepStream Detection & Segmentation Pipelines

**Two pipeline configurations** - object detection (s4) and semantic segmentation (s5):

| Pipeline | Model | Implementation | Input | Output | Purpose |
|----------|-------|----------------|-------|--------|---------|
| **s4** | YOLOv8n COCO | Pure C++ + Custom Parser | `in_s4` | `s4` | Object detection (80 COCO classes) with bounding boxes |
| **s5** | PeopleSemSegNet ONNX | Pure C++ + CUDA | `in_s5` | `s5` | Zero-copy GPU semantic segmentation overlay (green overlay on people) |

### Pipeline Order (CRITICAL)

```
nvurisrcbin (RTSP input)
  ↓
Bin with ghost pad
  ↓
nvstreammux (batch=1, 1280x720)
  ↓
nvinfer (ONNX → TensorRT)
  ↓
nvvidconv (NV12 → RGBA)          ← MUST come before nvosd
  ↓
nvosd (draw boxes, finalize segmentation metadata)
  ↓
rgba_caps (probe attachment point varies)
  ↓
nvvidconv_postosd (RGBA → I420)
  ↓
capsfilter (I420 format)
  ↓
nvv4l2h264enc (H.264 encoding)
  ↓
queue (frame buffering)          ← CRITICAL for smooth output
  ↓
h264parse
  ↓
rtspclientsink (RTSP output)
```

**Why this order matters:**
- **nvvidconv BEFORE nvosd**: nvosd requires RGBA format to draw (CPU mode)
- **queue after encoder**: Prevents choppy video by buffering frames

### C++ Probe Attachment

**Segmentation probe** attaches AFTER nvosd to access finalized metadata:

```cpp
// Attach probe to rgba_caps sink pad (after nvosd)
rgba_sinkpad = gst_element_get_static_pad(rgba_caps, "sink");
gst_pad_add_probe(rgba_sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
                  segmentation_probe_callback, NULL, NULL);
```

**Process flow:**
1. nvinfer outputs segmentation metadata
2. nvosd finalizes metadata (required!)
3. Probe accesses seg_meta → applies zero-copy GPU overlay

## Key Files

### main.cpp (207 lines)

Pure C++ GStreamer application - zero Python dependencies:

**Key Functions:**
- `bus_call()` (lines 39-61): Handles pipeline messages (EOS, errors)
- `on_pad_added()` (lines 18-37): Links nvurisrcbin dynamic pads to nvstreammux
- `main()` (lines 63-206): Creates pipeline, attaches probe, runs event loop

**nvurisrcbin Configuration:**
```cpp
g_object_set(G_OBJECT(source), "uri", rtsp_in, NULL);
g_object_set(G_OBJECT(source), "rtsp-reconnect-interval", 10, NULL);
g_object_set(G_OBJECT(source), "init-rtsp-reconnect-interval", 5, NULL);
g_object_set(G_OBJECT(source), "rtsp-reconnect-attempts", -1, NULL);  // Infinite
g_object_set(G_OBJECT(source), "select-rtp-protocol", 4, NULL);       // TCP-only
```

**Probe Attachment:**
```cpp
// Attach probe to rgba_caps sink pad (AFTER nvosd)
rgba_sinkpad = gst_element_get_static_pad(rgba_caps, "sink");
gst_pad_add_probe(rgba_sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
                  segmentation_probe_callback, NULL, NULL);
```

**Critical Optimizations:**
```cpp
g_object_set(G_OBJECT(encoder), "preset-id", 0, NULL);      // P1 (highest performance)
g_object_set(G_OBJECT(encoder), "profile", 2, NULL);        // Main profile
g_object_set(G_OBJECT(rtsp_sink), "latency", 200, NULL);    // 200ms buffer
```

### segmentation_probe_complete.cpp (299 lines)

C++ pad probe for zero-copy GPU segmentation overlay:

**Key Functions:**
- `segmentation_probe_callback()` (lines 67-281): Main probe callback
  - Gets GPU frame pointer from NvBufSurface
  - Accesses segmentation metadata (class_map on CPU)
  - GPU-only int→float conversion (zero CPU overhead)
  - Launches CUDA overlay kernel

**Zero-copy GPU processing:**
```cpp
// Get GPU frame pointer directly
NvBufSurface *surf = (NvBufSurface *)map_info.data;
void *frame_gpu_ptr = surf->surfaceList[frame_meta->batch_id].dataPtr;

// Get segmentation metadata
NvDsInferSegmentationMeta *seg_meta = (NvDsInferSegmentationMeta *)user_meta->user_meta_data;

// GPU-only int→float conversion (ZERO CPU overhead)
cudaMalloc((void**)&class_map_gpu, seg_size * sizeof(int));
cudaMemcpy(class_map_gpu, seg_meta->class_map, seg_size * sizeof(int), cudaMemcpyHostToDevice);

cudaMalloc((void**)&seg_float_gpu, seg_size * sizeof(float));
convert_classmap_gpu(class_map_gpu, seg_float_gpu, seg_size);  // GPU kernel

// Launch overlay kernel
launch_segmentation_overlay_direct(frame_gpu_ptr, seg_float_gpu, ...);
```

### segmentation_overlay_direct.cu (118 lines)

CUDA kernels for GPU-only processing:

**Kernels:**
1. `convert_int_to_float` (lines 69-74): GPU-only int→float conversion
2. `apply_segmentation_overlay_direct` (lines 9-66): Green overlay with alpha blending

**Key Features:**
- Input: RGBA frame (GPU), float segmentation data (GPU)
- Output: Green translucent overlay on person pixels (class 1)
- Thread grid: 16x16 blocks covering full frame resolution
- Alpha blending: `new = original * (1-alpha) + green * alpha`
- HWC layout support for segmentation masks

### up.sh

Pipeline orchestration script (currently runs s4 YOLO detection):

```bash
# s4: YOLOv8 COCO detection (80 classes including person)
docker run -d --name drishti-s4 --gpus all --rm --network host \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/config":/config \
  -v "$(pwd)/libnvdsinfer_custom_impl_Yolo.so":/app/libnvdsinfer_custom_impl_Yolo.so \
  ds_python:latest \
  /app/deepstream_app \
  rtsp://RELAY_IP:8554/in_s4 \
  rtsp://RELAY_IP:8554/s4 \
  /config/pgie_yolov8_coco.txt
```

**Volume Mounts:**
- `models/`: TensorRT engine cache + ONNX models (persistent across runs)
- `config/`: Inference configuration (can modify without rebuild, just restart)
- `libnvdsinfer_custom_impl_Yolo.so`: Custom YOLO parser library

### Dockerfile

Single-stage C++ build:
1. **Base**: DeepStream 8.0 Triton Multiarch
2. **Copy source**: main.cpp, segmentation_probe_complete.cpp, segmentation_overlay_direct.cu
3. **Build**: Runs build_app.sh to compile C++ application with CUDA kernels

**Key features:**
- Pure C++ (no Python runtime)
- CUDA kernels compiled with `nvcc`
- Links against DeepStream libraries (nvdsgst_meta, nvds_meta)
- Executable: `/app/deepstream_app`

### build_app.sh

Builds the C++ application:

```bash
# Compile CUDA kernel
nvcc -c segmentation_overlay_direct.cu -o segmentation_overlay_direct.o \
    --compiler-options '-fPIC' -arch=sm_75

# Compile C++ probe
g++ -c segmentation_probe_complete.cpp -o segmentation_probe_complete.o \
    -fPIC -I/opt/nvidia/deepstream/deepstream-8.0/sources/includes \
    -I/usr/local/cuda/include $(pkg-config --cflags gstreamer-1.0)

# Compile main application
g++ -c main.cpp -o main.o \
    -fPIC -I/opt/nvidia/deepstream/deepstream-8.0/sources/includes \
    $(pkg-config --cflags gstreamer-1.0 glib-2.0)

# Link into executable
g++ -o deepstream_app \
    main.o segmentation_probe_complete.o segmentation_overlay_direct.o \
    -L/usr/local/cuda/lib64 -L/opt/nvidia/deepstream/deepstream-8.0/lib \
    -lcudart -lnvdsgst_meta -lnvds_meta \
    $(pkg-config --libs gstreamer-1.0 glib-2.0)
```

### models/ (Git LFS)

**YOLO Detection (s4)**:
- `yolov8n.onnx` (13MB) - YOLOv8 nano object detection (80 COCO classes)
  - **Exported with**: DeepStream-Yolo custom export script (not standard Ultralytics)
  - **Source**: Ultralytics YOLOv8n converted with `DeepStream-Yolo/utils/export_yoloV8.py`
- `coco_labels.txt` - COCO class labels (person, bicycle, car, ...)

**Segmentation (s5)**: `peoplesemsegnet_vdeployable_shuffleseg_unet_onnx_v1.0.1/`
- `peoplesemsegnet_shuffleseg.onnx` (3.8MB) - Semantic segmentation (background, person)
- `labels.txt` - Label file (background, person)
- `peoplesemsegnet_shuffleseg_int8.txt` - INT8 calibration cache
- `pgie_unet_tlt_config_peoplesemsegnet_shuffleseg.txt` - NGC-provided reference config
- **Model source**: `nvidia/tao/peoplesemsegnet:deployable_shuffleseg_unet_onnx_v1.0.1` from NGC

**TensorRT cache** (generated, not in git):
- `*.engine` files - GPU-specific, ~4-5 minutes to generate on first run
- Stored in `/models/` via volume mount for persistence
- Note: YOLO engine currently saves to `/app/model_b1_gpu0_fp16.engine` (path mismatch issue)

## Development Workflow

### C++ Changes (requires rebuild)

C++ files are **compiled into the image**, rebuild required:

```bash
# Edit main.cpp, segmentation_probe_complete.cpp, or segmentation_overlay_direct.cu
vim segmentation_probe_complete.cpp

# Rebuild and restart
./build.sh && ./up.sh
```

### Config Changes (no rebuild)

Config files are **volume mounted**, just restart:

```bash
# Edit config file
vim config/pgie_peoplesemseg_onnx.txt

# Just restart container
docker restart drishti-s4
```

### Testing Changes

```bash
# Make changes to C++ source
./build.sh && ./up.sh

# Check logs
docker logs -f drishti-s4

# View output (WebRTC)
http://RELAY_IP:8889/s4/
```

## Performance Notes

**Current implementation**: Pure C++ with zero-copy GPU processing

**Performance metrics**:
- CPU usage: <5% (optimized from 100% in Python version)
- GPU processing: All segmentation overlay work on GPU
- Zero Python GIL overhead
- Single 720p stream with real-time segmentation and overlay

## Related Documentation

- **AGENTS.md**: AI agent guidelines, pipeline patterns, common pitfalls
- **STANDARDS.md**: Development standards, testing workflow
- **relay/README.md**: MediaMTX relay deployment guide
