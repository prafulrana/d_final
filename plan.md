# Current Plan

## Status: Cleanup and Optimization Phase

The Python single-stream implementation (`/root/d_final/`) is **fully functional** with robust RTSP reconnection using NVIDIA's nvurisrcbin built-in capabilities.

## Completed: nvurisrcbin-Based Reconnection ✓

**Final Solution** (working very well):
- ✓ Use nvurisrcbin (not uridecodebin) with built-in RTSP reconnection
- ✓ Wrap nvurisrcbin in GstBin with ghost pad (NVIDIA pattern from deepstream-test3)
- ✓ Configure reconnection properties:
  - `rtsp-reconnect-interval`: 10 seconds
  - `init-rtsp-reconnect-interval`: 5 seconds
  - `rtsp-reconnect-attempts`: -1 (infinite)
  - `select-rtp-protocol`: 4 (TCP-only, avoids 5-second UDP timeouts)
- ✓ Ghost pad pattern handles dynamic pad linking automatically
- ✓ Pipeline stays PLAYING, nvurisrcbin handles all reconnection internally
- ✓ Works with both clean disconnects (gst-launch) and abrupt closes (Larix)

**Key Insight**: TCP-only protocol (`select-rtp-protocol=4`) eliminates UDP timeout delays during reconnection attempts, enabling fast recovery.

## Completed: Fast Startup with Persistent Model Engine ✓

**Problem**: Engine rebuild took ~38 seconds on every restart.

**Solution**: Store model files in persistent `/models/` volume.
- ✓ Copied ONNX model to /models/
- ✓ Updated pgie_config.txt to use /models/ paths for both ONNX and engine
- ✓ Volume mounted in Docker run command
- ✓ Startup now ~5 seconds (instant engine load)

## Architecture

**Pipeline**: nvurisrcbin (in Bin + ghost pad) → nvstreammux → nvinfer → nvvideoconvert → nvdsosd → nvvideoconvert → capsfilter → nvv4l2h264enc → h264parse → rtspclientsink

**Key Properties**:
- Input: TCP-only RTSP via nvurisrcbin with automatic reconnection
- Output: TCP-only RTSP via rtspclientsink (`protocols=0x00000004`)
- Inference: TrafficCamNet (resnet18_trafficcamnet_pruned.onnx)
- Container: Named "drishti", detached mode, persistent /models volume

## Reference Files

- `/root/d_final/app.py` - Main application (318 lines, clean implementation)
- `/root/d_final/Dockerfile` - DeepStream 8.0 with Python bindings
- `/root/d_final/up.sh` - Build and run script with named container cleanup
- `/root/deepstream-8.0/sources/apps/sample_apps/deepstream-test3/` - NVIDIA reference for ghost pad pattern
- `/root/deepstream-8.0/sources/gst-plugins/gst-nvurisrcbin/` - nvurisrcbin source code

---

## Archived: Manual Reconnection Attempts (Abandoned)

**Why Abandoned**: nvurisrcbin's built-in reconnection is more robust and simpler than manual `stop_release_source()` + timer-based re-add pattern.

**Previous Attempts**:
- Stream-eos detection with manual source bin removal/recreation (deadlock issues)
- Watchdog-based reconnection monitoring MediaMTX API (overcomplicated)
- Poll loop conflicts with state_lock during re-add operations

**Lesson Learned**: Always research NVIDIA's built-in capabilities before implementing custom solutions.
