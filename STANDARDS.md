# Development Standards

## Build

```bash
./up.sh
```

This will:
1. Build Docker image: `ds_python:latest`
2. Clean up old containers named "drishti"
3. Kill orphaned containers using the image
4. Run in detached mode with GPU access and persistent /models volume

**Build Time**: ~2 minutes (first build), ~5 seconds (cached)

## Run

Container runs automatically in detached mode via `up.sh`. Logs available via:

```bash
docker logs -f drishti
```

**Startup Time**: ~5 seconds (with cached TensorRT engine in /models)

## Test

### Manual Test - Initial Connection

```bash
# 1. Start container
./up.sh

# 2. Publish RTSP stream to input
# From Larix mobile app:
rtsp://34.100.230.7:8554/in_s0

# Or from gst-launch:
gst-launch-1.0 videotestsrc ! x264enc ! rtspclientsink location=rtsp://34.100.230.7:8554/in_s0

# 3. View output stream with detection boxes
ffplay -rtsp_transport tcp rtsp://34.100.230.7:8554/s0

# 4. Check logs for "Pipeline is PLAYING"
docker logs drishti | grep PLAYING
```

### Manual Test - Reconnection

```bash
# While stream is running:
# 1. Stop publishing to in_s0 (close Larix, Ctrl+C gst-launch)
# 2. Wait ~10 seconds
# 3. Check logs show "Resetting source -1, attempts: 1"
docker logs drishti | tail -20

# 4. Restart publishing to in_s0
# 5. Verify output stream recovers automatically
# 6. Check logs show "In cb_newpad" indicating successful reconnection
docker logs drishti | grep "cb_newpad"
```

**Expected Behavior**:
- Initial connection: ~5 seconds to PLAYING state
- Reconnection after disconnect: ~10-15 seconds (rtsp-reconnect-interval=10)
- No UDP timeout delays (TCP-only mode)
- Works with both clean disconnects (gst-launch Ctrl+C) and abrupt closes (Larix app close)

## Clean

```bash
docker rm -f drishti
docker rmi ds_python:latest  # Optional: remove image
```

## Debug

### Enable Verbose GStreamer Logging

Edit `Dockerfile`:
```dockerfile
ENV GST_DEBUG=3  # Change from 2 to 3 or 4
```

Then rebuild:
```bash
./up.sh
```

**GST_DEBUG Levels**:
- 0: None
- 1: Errors only
- 2: Warnings + Errors (current)
- 3: Info + Warnings + Errors
- 4: Debug (very verbose)

### Check Reconnection Properties

```bash
docker exec -it drishti gst-inspect-1.0 nvurisrcbin | grep -A3 "rtsp-reconnect"
```

### Check Pipeline State

```bash
# Should show "Current state 4" (PLAYING)
docker logs drishti | grep "Current state"
```

### Common Issues

**Issue**: Stream not reconnecting after disconnect
- **Check**: Logs show "Could not receive any UDP packets for 5.0000 seconds"
- **Cause**: Missing `select-rtp-protocol=4` (TCP-only)
- **Fix**: Verify line 91 in app.py has TCP-only setting

**Issue**: TensorRT engine rebuild on every restart
- **Check**: Logs show "Building TensorRT Engine" taking 30+ seconds
- **Cause**: /models volume not mounted or engine file missing
- **Fix**: Verify `docker run -v $PWD/models:/models` in up.sh

**Issue**: "Failed to link decoder src pad to source bin ghost pad"
- **Check**: Logs show "Error: Decodebin did not pick nvidia decoder plugin"
- **Cause**: NVMM memory features not found (GPU issue or wrong decoder)
- **Fix**: Verify `--gpus all` flag and NVIDIA driver installed

## Code Style

- Python 3
- 4-space indentation
- Max line length: 100 characters
- Follow NVIDIA DeepStream sample patterns
- Use sys.stderr for errors, print() for info

## Property References

### nvurisrcbin Critical Properties

```python
# Reconnection (app.py lines 88-91)
uri_decode_bin.set_property("rtsp-reconnect-interval", 10)         # Seconds between retries
uri_decode_bin.set_property("init-rtsp-reconnect-interval", 5)     # Initial delay
uri_decode_bin.set_property("rtsp-reconnect-attempts", -1)         # -1 = infinite
uri_decode_bin.set_property("select-rtp-protocol", 4)              # 4 = TCP-only
```

**RTP Protocol Values** (from nvurisrcbin source):
- 1: UDP only
- 2: UDP multicast only
- 4: TCP only (recommended)
- 7: UDP + UDP-mcast + TCP (default, causes 5-second UDP timeouts)

### nvstreammux Properties

```python
# app.py lines 222-226
g_streammux.set_property("width", 1280)
g_streammux.set_property("height", 720)
g_streammux.set_property("batch-size", 1)
g_streammux.set_property("batched-push-timeout", 33000)  # 33ms
g_streammux.set_property("live-source", 1)
```

### rtspclientsink Properties

```python
# app.py lines 218-219
rtsp_sink.set_property("location", rtsp_out)
rtsp_sink.set_property("protocols", 0x00000004)  # TCP-only (matches input)
```

## Commit Guidelines

When making changes:

1. Test reconnection after every modification
2. Verify TensorRT engine caching still works
3. Check startup time remains <10 seconds
4. Update relevant docs (plan.md, STRUCTURE.md, AGENTS.md)

**Good Commit Examples**:
- `app: configure TCP-only protocol to eliminate UDP timeouts`
- `dockerfile: persist ONNX model in /models for engine caching`
- `up.sh: use named container to prevent duplicates`

**Bad Commit Examples**:
- `fixed stuff` (too vague)
- `updated app.py` (what changed?)
- `trying to fix reconnection` (implies broken state)
