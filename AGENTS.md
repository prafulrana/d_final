# Repository Guidelines

This repo hosts a **Python-based single-stream DeepStream pipeline** that processes RTSP input with inference and outputs to RTSP via rtspclientsink. It uses nvurisrcbin for automatic RTSP reconnection.

## Project Structure & Modules
- `app.py` — Main Python application: nvurisrcbin → nvstreammux → nvinfer → OSD → encoder → rtspclientsink
- `Dockerfile` — DeepStream 8.0 Python environment with TensorRT engine caching
- `up.sh` — Build and run script (creates named container `drishti`)
- `models/` — Persistent TensorRT engine cache (volume mounted)
- `relay/` — MediaMTX relay server configuration (GCP VM). Default zone: `asia-south1-c`
- `pgie_config.txt` — nvinfer configuration for TrafficCamNet ONNX model
- `STANDARDS.md`, `STRUCTURE.md` — Build/test/debug documentation

## Build, Test, Run
- Build & Run: `./up.sh` (builds `ds_python:latest`, runs as `drishti` container)
- View logs: `docker logs -f drishti`
- Publish test stream: Use Larix app or `gst-launch-1.0` to `rtsp://server:8554/in_s0`
- View output: `ffplay -rtsp_transport tcp rtsp://server:8554/s0`
- Relay deploy: `cd relay && terraform init && terraform apply -var project_id=<GCP_PROJECT>`

## Coding Style & Conventions
- Python 3 with GStreamer bindings (`gi.repository.Gst`)
- 4-space indentation, max 100 character lines
- Follow NVIDIA DeepStream Python sample patterns
- Use `sys.stderr.write()` for errors, `print()` for info
- Keep functions focused with early returns

## Testing Guidelines
- **Initial connection test**: Publish to `in_s0`, verify output at `s0`, check logs for "Pipeline is PLAYING"
- **Reconnection test**: Stop publisher, wait ~10s for reconnect logs, restart publisher, verify auto-recovery
- **Video quality test**: Compare smoothness of `in_s0` vs `s0` - output should match input quality
- **Performance test**: Verify startup time <10s with cached TensorRT engine
- Include logs and minimal repro for any pipeline or encoder changes

## Commits & Pull Requests
- Commit style: `scope: imperative summary` (e.g., `app: add queue element for smooth RTSP output`)
- PRs must include: change rationale, test steps (especially video quality), and doc updates when behavior changes
- Always test reconnection after modifications

## RTSP Reconnection Pattern (Python) - nvurisrcbin Method

**RECOMMENDED**: For Python DeepStream applications with RTSP sources, use **nvurisrcbin** with built-in reconnection:

1. **Use nvurisrcbin (not uridecodebin)**:
   ```python
   uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
   uri_decode_bin.set_property("uri", rtsp_uri)
   ```

2. **Configure reconnection properties**:
   ```python
   uri_decode_bin.set_property("rtsp-reconnect-interval", 10)  # Seconds between retries
   uri_decode_bin.set_property("init-rtsp-reconnect-interval", 5)  # Initial retry interval
   uri_decode_bin.set_property("rtsp-reconnect-attempts", -1)  # Infinite retries
   uri_decode_bin.set_property("select-rtp-protocol", 4)  # TCP-only (avoids UDP timeouts)
   ```

3. **Wrap in Bin with ghost pad** (NVIDIA pattern from deepstream-test3):
   ```python
   nbin = Gst.Bin.new("source-bin-00")
   Gst.Bin.add(nbin, uri_decode_bin)
   bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))

   # Connect pad-added to set ghost pad target when decoder pad appears
   uri_decode_bin.connect("pad-added", cb_newpad, nbin)
   ```

4. **cb_newpad sets ghost pad target**:
   ```python
   def cb_newpad(decodebin, decoder_src_pad, source_bin):
       if features.contains("memory:NVMM"):
           bin_ghost_pad = source_bin.get_static_pad("src")
           bin_ghost_pad.set_target(decoder_src_pad)
   ```

**Key Advantages**:
- No manual reconnection logic needed
- Pipeline stays PLAYING throughout
- nvurisrcbin handles all disconnects internally
- TCP-only avoids 5-second UDP timeout delays
- Works with both clean disconnects and abrupt closes

**Reference**: See `/root/d_final/app.py` for complete working implementation

## Notes for Agents
- Keep edits within the repo root; align with `STRUCTURE.md` and `STANDARDS.md`
- Avoid new frameworks; prefer surgical changes to `app.py`
- **DO NOT delete researched settings on a whim** - if a configuration was researched and implemented, validate thoroughly before removing
- **Research first, change second** - especially for encoder/timing/buffer settings
- GCP authentication: If running as root but gcloud/terraform are in prafulrana's home, check `gcloud auth list` FIRST before attempting application-default login. If an account is already authenticated, use `gcloud auth print-access-token` to get a token for Terraform. Export PATH: `export PATH="/usr/bin:/bin:/usr/local/bin:/home/prafulrana/google-cloud-sdk/bin"`
- Terraform on Ubuntu: Install from HashiCorp repo: `wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg && echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com noble main" | tee /etc/apt/sources.list.d/hashicorp.list && apt-get update && apt-get install -y terraform`

## Common Pitfalls & Checks
- **Missing queue elements cause choppy video** - Always include queue after encoder for frame buffering
- **Encoder optimization is critical** - Use `preset-id=0` (P1 highest performance) and `profile=2` (Main)
- **rtspclientsink needs latency** - Set `latency=200` (ms) for smooth RTP timing, otherwise warnings about "can't determine running time"
- **Don't confuse lag with choppiness** - Lag is latency (seconds), choppy is frame stuttering (buffering/timing issue)
- **TCP-only for both input/output** - `select-rtp-protocol=4` on nvurisrcbin, `protocols=0x00000004` on rtspclientsink
- **TensorRT engine caching** - Mount `/models` volume to avoid 30+ second rebuilds on every restart
- **Ensure NVMM memory** - Check `cb_newpad` logs for "memory:NVMM" features, otherwise GPU decode failed

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
