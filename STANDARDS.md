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
# app.py lines 233-237
g_streammux.set_property("width", 1280)
g_streammux.set_property("height", 720)
g_streammux.set_property("batch-size", 1)
g_streammux.set_property("batched-push-timeout", 4000000)  # 4 seconds for live RTSP
g_streammux.set_property("live-source", 1)
```

**Critical**: `batched-push-timeout` must be 4 seconds (4000000 microseconds) for live RTSP sources to prevent "reader too slow" errors from MediaMTX. Lower values (33ms) cause MediaMTX to drop hundreds of frames during pipeline startup/reconnection and tear down the session.

### nvv4l2h264enc Encoder Properties

```python
# app.py lines 201-205
encoder.set_property("bitrate", 3000000)
encoder.set_property("profile", 2)          # 2 = Main profile (better compression/smoothness)
encoder.set_property("preset-id", 0)        # 0 = P1 (highest performance preset)
encoder.set_property("insert-sps-pps", 1)
encoder.set_property("iframeinterval", 30)
```

**Profile Values**:
- 0: Baseline (low complexity, mobile devices)
- 2: Main (better compression, recommended for streaming)
- 4: High (highest compression, more CPU overhead)

**Preset-ID Values** (P1-P7):
- 0: P1 (highest performance, lowest latency)
- 6: P7 (lowest performance, highest compression)

### Queue Element (Required for Smooth Output)

```python
# app.py lines 207-212
queue = Gst.ElementFactory.make("queue", "queue")
# Place AFTER encoder, BEFORE h264parse
```

**Why Critical**: Without queue, frame timing issues cause choppy/stuttering output. Queue provides buffering that smooths frame delivery to rtspclientsink.

### rtspclientsink Properties

```python
# app.py lines 228-230
rtsp_sink.set_property("location", rtsp_out)
rtsp_sink.set_property("protocols", 0x00000004)  # TCP-only (matches input)
rtsp_sink.set_property("latency", 200)           # 200ms buffer for smooth RTP timing
```

**Critical**: `latency=200` eliminates "Can't determine running time for this packet" warnings and ensures smooth RTP packet timing. Without it, output appears choppy despite correct frame rate.

## Performance Troubleshooting

**Before you change resolution, disable inference, or drop frames**:

1. **Remember the baseline**: This system handles 64 concurrent 1080p streams with inference
2. **Check configuration first**: 99% of "performance" issues are buffer/timeout misconfigurations
3. **Verify the actual bottleneck**: Use `nvidia-smi`, `GST_DEBUG=3`, network analysis

**Common mistakes when debugging "slow" pipelines**:
- ❌ Assuming resolution is too high → It's not, we handle 1080p @ 64x
- ❌ Disabling inference to "test" → Masks the real issue, defeats purpose
- ❌ Reducing bitrate → Encoding is GPU-accelerated, not the bottleneck
- ❌ Dropping frames at source → Doesn't fix timing issues

**Actual fixes (in order of likelihood)**:
1. **Add queue element** after encoder (choppy video = missing queue 90% of the time)
2. **Set encoder preset-id=0** (P1 highest performance) and profile=2 (Main)
3. **Set rtspclientsink latency=200** (fixes "can't determine running time" warnings)
4. Increase `batched-push-timeout` to 4000000 (4 seconds)
5. Check TensorRT engine is cached (not rebuilding every run)
6. Verify NVMM memory is being used (check `cb_newpad` logs)
7. Check network latency/packet loss to RTSP server
8. Only then: Profile actual GPU/CPU usage

## Video Quality Troubleshooting

**Symptom**: Output stream is choppy/stuttering while input is smooth

**Root Cause (99% of the time)**: Missing queue element or encoder misconfiguration

**Solution**:
1. **Verify queue element exists** in pipeline: `encoder → queue → h264parse`
2. **Check encoder properties**:
   - `preset-id=0` (P1, not default)
   - `profile=2` (Main, not Baseline)
3. **Check rtspclientsink latency**:
   - Must be set to 200ms minimum
   - Logs show "can't determine running time" if missing
4. **Compare with vanilla deepstream-app**: If vanilla is smooth with same source, it's our config

**NOT the issue** (don't waste time here):
- Resolution too high (we handle 64x 1080p streams)
- Bitrate too high (GPU encoding is fast)
- Network lag (that's latency, not choppiness)
- MediaMTX relay configuration (if in_s0 is smooth, relay is fine)

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
