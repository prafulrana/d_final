# Standards

## Always Read DeepStream Samples First
- Before making any code or doc changes, read the samples under `deepstream-8.0/` in this repo. Align implementation and configuration with DeepStream 8.0 patterns, especially:
  - `sources/apps/apps-common/src/deepstream_sink_bin.c` (RTSP via UDP-wrap, `rtph264pay` + `udpsink`, `udpsrc name=pay0`)
  - `sources/apps/sample_apps/deepstream-app` (demux/sink wiring and config parsing)
  - `samples/configs/deepstream-app*` (`rtsp-port`, sink settings, demux usage)
- Do not introduce new approaches that diverge from these patterns unless documented rationale is added here.

## How To Run
- Requirements: Docker with NVIDIA runtime (`--gpus all`), access to nvcr.io.
- Build: `./build.sh`
- Run: `./run.sh` (defaults to 2 outputs: `/s0`, `/s1`)
- Quick test from macOS:
  - `/test` (sanity): `ffplay -rtsp_transport tcp rtsp://<host>:8554/test`
  - `/s0`: `ffplay -rtsp_transport tcp rtsp://<host>:8554/s0`
  - `/s1`: `ffplay -rtsp_transport tcp rtsp://<host>:8554/s1`

## Configuration Strategy
- Single config file: `pipeline.txt` defines all pre‑demux behavior and sources.
- C code only handles post‑demux and RTSP. No encoder toggles or branches.
 - Per‑stream encode uses `nvv4l2h264enc` (NVENC), then RTP/UDP egress to localhost; RTSP wraps from UDP (DeepStream pattern).
 - Important: `pipeline.txt` must be a single line (no comments or shell line continuations). It is parsed by `gst_parse_launch`, not a shell.
 - OSD overlays are enabled by default (`USE_OSD=1`), matching DeepStream samples. Disable only for scale testing.
 - Queue per branch is tuned for low latency: `leaky=2` (downstream) and `max-size-time=200ms`.
 - RTSP factories wrap UDP using `udpsrc port=<p> buffer-size=524288 name=pay0` with H264 RTP caps.

## Minimal Env Vars
- `STREAMS` — number of `/sN` endpoints to expose (default 2)
- `RTSP_PORT` — RTSP TCP port (default 8554; auto‑increments if busy)
- `BASE_UDP_PORT` — starting UDP port for per‑stream RTP egress (default 5000)

## Code Cleanliness
- Favor config strings (pipeline, encoder choices) over code. Keep C small.
- Do not mix unrelated concerns; post‑demux logic only.
- Prefer explicit pad requests and clear error logs for linking.
- Keep defaults sensible; make behavior tunable via envs.
- No Python in this app; avoid extra frameworks.

## Troubleshooting
- `/test` works but `/sN` returns 503:
  - Confirm logs show: `Linked demux src_N to UDP egress port 5000+N` and `RTSP mounted ... (udp-wrap H264 RTP @127.0.0.1:5000+N)`.
  - Ensure `pipeline.txt` has at least N+1 URIs and `max-batch-size` adequate.
  - Rebuild and rerun if you changed `pipeline.txt`.
- Port conflicts:
  - RTSP retries 8554..+9. Use the logged port in your ffplay URL.
  - DeepStream’s REST (9000) may already be bound; it doesn’t affect RTSP.

## Scaling Considerations
- Match `STREAMS` with `max-batch-size` in `pipeline.txt`; set framerate/resize pre‑demux for uniform NVENC input.
- Bandwidth planning: e.g., 64× streams at ~3 Mbps ≈ 192 Mbps aggregate.
- Encoder settings: keep `insert-sps-pps=1` and IDR/I‑frame intervals aligned to framerate; default bitrate ~3 Mbps @ 720p30.

## Style
- C: consistent naming, early returns, minimal globals, no dead paths.
- One config until stable: avoid options creep until `/sN` are reliable.
- Patches: keep changes surgical; avoid unrelated edits.
- Docs: concise, task‑oriented, and aligned to actual code.
