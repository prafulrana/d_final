# Repository Guidelines

This repo hosts a C-based GStreamer/DeepStream pipeline that pushes H.264 over RTSP directly (rtspclientsink). The pre‑demux pipeline is defined via `pipeline.txt` (config-driven); post‑demux encoding and RTSP push are built in C.

## Project Structure & Modules
- `pipeline.txt` — Pre‑demux: `nvmultiurisrcbin → nvinfer (pgie.txt) → nvstreamdemux name=demux` with `uri-list=...`.
- `src/` — C sources: `main.c` (entry), `app.c` (lifecycle + pipeline), `branch.c` (per‑stream), `control.c` (HTTP), `config.c` (args/env), headers.
- `Dockerfile`, `build.sh`, `run.sh`, `sanity.sh`, `STANDARDS.md`, `STRUCTURE.md`, `pgie.txt`.
- `relay/` — Terraform IaC for the MediaMTX relay VM (GCP). Default zone: `asia-south1-c`.
- Engine cache persists under `./models` (mounted by `run.sh`).

## Build, Test, Run
- Build: `./build.sh` (includes `sanity.sh`).
- Run: `./run.sh` (envs: `RTSP_PUSH_URL_TMPL`, `CTRL_PORT`, `HW_THRESHOLD`, `SW_MAX`, `MAX_STREAMS`).
  - Ensures a single sender container (`drishti`) is running.
  - By default publishes two streams (s0,s1) to the new relay.
- Control API: `curl http://localhost:8080/status` and `curl http://localhost:8080/add_demo_stream` (optional adds beyond `uri-list`).
- Push target defaults to `rtsp://34.14.144.178:8554/s0` (override with `RTSP_PUSH_URL_TMPL`).
- Relay deploy: `cd relay && terraform init && terraform apply -var project_id=<GCP_PROJECT>`.

## Coding Style & Conventions
- C with GLib/GStreamer. Indent 2 spaces; 80–100 cols. File names `lower_snake_case.c/.h`; symbols `snake_case`; globals prefixed `g_`.
- Use logging macros in `log.h` (`LOG_ERR/WRN/INF`). Keep functions short with early returns.
- Do not expand pre‑demux in code—prefer `pipeline.txt`. Post‑demux (branch/encode/RTSP push) stays in C.

## Testing Guidelines
- Sanity inside container: `docker run --rm -i batch_streaming:latest bash -s < sanity.sh`.
- End‑to‑end: edit `pipeline.txt` URIs, `./run.sh`, confirm `/status`; inspect the remote RTSP path for the pushed stream.
- Include logs and minimal repro for any pipeline or encoder changes.

## Commits & Pull Requests
- Commit style: `Scope: imperative summary` (e.g., `Server: bootstrap branches from uri-list`).
- PRs must include: change rationale, test steps, configs/envs, and doc updates when behavior changes (`STANDARDS.md`, `STRUCTURE.md`).

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
- Keep edits within the repo root; align with `STRUCTURE.md` and `STANDARDS.md`.
- Avoid new frameworks; prefer small, surgical changes to `app.c`, `branch.c`, `control.c`, `config.c`.
- GCP authentication: If running as root but gcloud/terraform are in prafulrana's home, check `gcloud auth list` FIRST before attempting application-default login. If an account is already authenticated, use `gcloud auth print-access-token` to get a token for Terraform. Export PATH: `export PATH="/usr/bin:/bin:/usr/local/bin:/home/prafulrana/google-cloud-sdk/bin"`.
- Terraform on Ubuntu: Install from HashiCorp repo: `wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg && echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com noble main" | tee /etc/apt/sources.list.d/hashicorp.list && apt-get update && apt-get install -y terraform`.

## Common Pitfalls & Checks
- Ensure remote RTSP host is reachable from the container host.
- Encoder props: `nvv4l2h264enc` varies by platform. Guard property sets with `g_object_class_find_property` (no hard-coded `maxperf-enable`).
- Startup order: create/mount branches before setting pipeline to `PLAYING` to avoid "data flow before segment" warnings.
- File inputs: in `pipeline.txt`, prefer `sync-inputs=true` and a larger `batched-push-timeout` for file URIs; align `pgie` batch-size/engine with `uri-list` or expect an engine rebuild.

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
