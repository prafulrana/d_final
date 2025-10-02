# DeepStream Batch Inference Multi-Stream RTSP Server

**Milestone Achievement:** Efficient batch inference with individual stream outputs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  nvmultiurisrcbin (30 video streams)                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  nvinfer (BATCH-30: Single GPU call for all 30 streams)        │
│  TensorRT FP16, ResNet18 Traffic Detection                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  nvstreamdemux (Split batch into individual streams)            │
└─────────────┬───────────────────────────────┬───────────────────┘
              │                               │
              ▼                               ▼
      ┌───────────────┐             ┌───────────────┐
      │  Stream 0     │             │  Stream 1     │
      │  • queue      │             │  • queue      │
      │  • nvosdbin   │             │  • nvosdbin   │
      │  • NVENC H264 │             │  • NVENC H264 │
      │  • RTP/UDP    │             │  • RTP/UDP    │
      └───────┬───────┘             └───────┬───────┘
              │                             │
              └─────────────┬───────────────┘
                            │
                            ▼
      ┌─────────────────────────────────────────┐
      │  C RTSP Server (GstRtspServer)          │
      │  • Wraps UDP (udpsrc pay0, H264 RTP)    │
      │  • rtsp://IP:8554/s0                    │
      │  • rtsp://IP:8554/s1                    │
      └─────────────────────────────────────────┘
```

## Key Features

- **Single Pipeline**: All streams processed in one GStreamer pipeline
- **Batch Inference**: 30x efficiency - AI runs once for all 30 streams
- **Per-Stream OSD**: Detection overlays applied post-demux (no blinking)
- **Dynamic Capable**: nvmultiurisrcbin supports runtime stream add/remove
- **Minimal Code**: Pre-demux via config string; C post-demux + RTSP only

## Files

- `src/main.c` - Tiny entrypoint (calls setup/loop/teardown)
- `src/app.c` - Pipeline + RTSP server setup; lifecycle (builds DS pre-demux in code: nvmultiurisrcbin → nvinfer(pgie.txt) → nvstreamdemux)
- `src/branch.c` - Per-stream branch building
- `src/control.c` - Simple HTTP control (`/add_demo_stream`, `/status`)
- `src/config.c` - Config parsing and helpers
- `src/log.h`, `src/state.h` - Logging macros and shared state
- `pipeline.txt` - Optional pre-demux pipeline (compatibility)
- `pgie.txt` - TensorRT inference configuration
- `Dockerfile` - Container image
- `build.sh` - Build script
- `run.sh` - Run script

## Usage

### Build
```bash
./build.sh
```

### Run
```bash
./run.sh
```


### Access Streams
```bash
# Stream 0 with AI detections
ffplay -rtsp_transport tcp rtsp://10.243.223.217:8554/s0

# Stream 1 with AI detections
ffplay -rtsp_transport tcp rtsp://10.243.223.217:8554/s1
```

## Performance

- **GPU**: Single TensorRT inference call for all 30 streams
- **Latency**: ~100ms (includes decode, inference, encode)
- **Throughput**: 30 streams @ 30fps = 900 FPS total
- **Memory**: Shared batch buffers, zero-copy demux

## Configuration

- DeepStream stage is built in code: `nvmultiurisrcbin → nvinfer (pgie.txt) → nvstreamdemux`, and per‑stream overlays use `nvosd` (nvosdbin) after demux.
- Minimal envs:
  - Removed. Server starts empty and uses control API to add streams.
- `RTSP_PORT` — RTSP TCP port (default 8554)
- `CTRL_PORT` — Control HTTP port for `/add_demo_stream` and `/status` (default 8080; fixed for Docker HEALTHCHECK)
- `HW_THRESHOLD` — Number of NVENC (HW) streams to allow before using SW encoders (default 8)
- `SW_MAX` — Number of software-encoded streams allowed (default 56)
- `MAX_STREAMS` — Override total capacity (default `HW_THRESHOLD + SW_MAX`)
- `X264_THREADS` — Threads per x264 encoder (default ~half cores, capped at 4)
 
Output rate
- RTSP pacing is real-time via sink clock: we set `udpsink sync=true` so encoded output does not run faster than realtime (e.g., 30 fps sources play at 30 fps).

## Future Enhancements

- [ ] REST API for dynamic stream control
- [ ] WebRTC output option
- [ ] Multi-model cascade (SGIE)
- [ ] Cloud storage integration
- [ ] Metrics/monitoring dashboard

## Technical Notes

### Why This Architecture?

1. **Batch Efficiency**: GPU processes 30 streams in ~same time as 1 stream
2. **Clarity**: DeepStream stage defined in code (nvinfer + nvosd), simple control API
3. **Stable OSD**: Per-stream OSD avoids batch synchronization issues
4. **Production Ready**: Uses battle-tested deepstream-app components

### Challenges Overcome

- ✅ Batch inference with individual outputs (demux post-inference)
- ✅ Non-blinking detection overlays (per-stream OSD)
- ✅ Efficient memory usage (shared buffers, zero-copy)
- ✅ Clean separation (pipeline vs serving layer)
- ✅ Startup sanity checks for required plugins; clear error messages
- ✅ Docker HEALTHCHECK via control `/status` (requires `CTRL_PORT`)

## References

- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [nvmultiurisrcbin Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmultiurisrcbin.html)
- [GStreamer RTSP Server](https://gstreamer.freedesktop.org/documentation/gst-rtsp-server/)

---

**Date:** 2025-09-30
**Status:** ✅ x264enc (CPU) + UDP-wrapped RTSP (C server)
**Version:** 1.2
