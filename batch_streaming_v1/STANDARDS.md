# Standards

## Always Read DeepStream Samples First
- Before making any code or doc changes, read the samples under `deepstream-8.0/` in this repo. Align implementation and configuration with DeepStream 8.0 patterns, especially:
  - `sources/apps/apps-common/src/deepstream_sink_bin.c` (RTSP via UDP-wrap, `rtph264pay` + `udpsink`, `udpsrc name=pay0`)
  - `sources/apps/sample_apps/deepstream-app` (demux/sink wiring and config parsing)
  - `samples/configs/deepstream-app*` (`rtsp-port`, sink settings, demux usage)
- Do not introduce new approaches that diverge from these patterns unless documented rationale is added here.

## How To Run (Single Happy Path — C, master)
- Requirements: Docker with NVIDIA runtime (`--gpus all`), access to nvcr.io.
- Build: `./build.sh`
- Start service (empty, only `/test` mounted): `./run.sh`
- Add one demo stream (control API): `curl http://localhost:8080/add_demo_stream`
  - Response (example): `{ "path": "/s0", "url": "rtsp://<host>:8554/s0" }`
  - Repeat to add `/s1`, `/s2`, ... up to capacity. Default hard limits: first 8 NVENC (HW), then software up to 56 (total 64). Override via envs below.
- Play from macOS: `ffplay -rtsp_transport tcp rtsp://<host>:8554/s0`
- Sanity any time: `ffplay -rtsp_transport tcp rtsp://<host>:8554/test`

## Configuration Strategy (C, master)
- Pre‑demux (DeepStream) is built in code: `nvmultiurisrcbin → nvinfer (pgie.txt) → nvstreamdemux name=demux`, and per‑stream overlays are applied with `nvosd` (nvosdbin) after demux.
- C code handles post‑demux and RTSP. Keep options minimal.
- Per‑stream encode policy (hard limits for predictability):
  - First 8 streams: try NVENC (`nvv4l2h264enc`) with low‑latency props.
  - Remaining streams: software H.264 (`x264enc` → fallback `avenc_h264` → `openh264enc`) including NVMM→I420 CPU hop.
  - Egress: RTP/UDP to localhost; RTSP wraps from UDP (DeepStream pattern).
 - RTSP caps: advertise payload type in caps. Do not set `pt` on `udpsrc` (set `payload=96` in caps and/or `pt` on the payloader).
- PGIE configuration comes from `pgie.txt`, and OSD uses `nvosd` (nvosdbin) on each stream.
 - Queue per branch is tuned for low latency: `leaky=2` (downstream) and `max-size-time=200ms`.
 - RTSP factories wrap UDP using `udpsrc port=<p> buffer-size=524288 name=pay0` with H264 RTP caps.
 - Control API: service starts with no `/sN`. Hitting `GET /add_demo_stream` auto‑adds a sample source and mounts the next `/sN`, returning its RTSP URL as JSON.
   - Capacity: hard limit defaults — `HW_THRESHOLD=8`, `SW_MAX=56`, `MAX_STREAMS=64`. Override via envs.
   - Requests beyond capacity return HTTP 429 with `{ "error": "capacity_exceeded", "max": <N> }`.
   - Health: `GET /status` returns current capacity and per‑stream encoder type.
 - Add sources at runtime via DeepStream REST (9000/9010) triggered by the control API.

## Encoder Options and Limits
- Default: NVENC for the first `HW_THRESHOLD` streams; software encoders beyond that.
- Known limit: Hardware encoder session count is finite and device‑dependent. Hitting the limit typically surfaces as:
  - `nvv4l2h264encX: Device '/dev/v4l2-nvenc' failed during initialization ... S_FMT failed`
  - Cascading `not-negotiated (-4)` warnings in downstream elements (e.g., `nvinfer`, `qtdemux`).
- If you switch to NVENC (`nvv4l2h264enc`), be aware of session limits by GPU/driver.

### Software Encoder (CPU)
- Packages (already installed in Dockerfile): `gstreamer1.0-plugins-ugly`, `gstreamer1.0-libav`, `libx264-164`.
  - Verify: `gst-inspect-1.0 x264enc` (or fallback `avenc_h264`).
Already implemented by default:
- Chain: `… NVMM RGBA → nvosdbin → NVMM NV12 → [NVENC | NVMM→I420] → encoder → h264parse → rtph264pay → udpsink`.
- x264enc props: `tune=zerolatency, speed-preset=ultrafast, bitrate=3000, key-int-max=60, bframes=0, threads≈half cores (cap 4)`; override with `X264_THREADS`.
- Note: CPU cost increases with streams. Adjust bitrate/fps/resolution as needed.

## Environment Variables (C, master)
- `RTSP_PORT` — RTSP TCP port (default 8554; auto‑increments if busy)
- `BASE_UDP_PORT` — starting UDP port for per‑stream RTP egress (default 5000)
- `SAMPLE_URI` — demo URI used by `add_demo_stream` (default DS sample 1080p H.264)
- `PUBLIC_HOST` — host/IP to return in RTSP URLs (default 127.0.0.1; override in run.sh)
- `CTRL_PORT` — HTTP control port (default 8080; Docker HEALTHCHECK uses this)
- `HW_THRESHOLD` — number of NVENC streams before switching to software (default 8)
- `SW_MAX` — number of software streams allowed (default 56)
- `MAX_STREAMS` — total allowed streams (default `HW_THRESHOLD + SW_MAX`)
- `X264_THREADS` — threads per x264 encoder (default ~half cores, capped at 4)

## Engine Caching
- The TensorRT engine is serialized to the host at `./models/...engine` (mounted into the container at `/models`).
- First run builds the engine (batch-64) and subsequent runs reuse it automatically.
- To force a rebuild, delete the engine file and rerun `./run.sh`.

## Code Cleanliness
- Favor config strings (pipeline, encoder choices) over code. Keep C small.
- Do not mix unrelated concerns; post‑demux logic only.
- Prefer explicit pad requests and clear error logs for linking.
- Keep defaults sensible; make behavior tunable via envs.
- No Python in this app; avoid extra frameworks.
- Use the logging macros for consistency: `LOG_ERR`, `LOG_WRN`, `LOG_INF`.
- Keep functions short; use the 4–5 file layering:
  - `main.c` — tiny entrypoint
  - `app.c` — setup/loop/teardown + RTSP server
  - `branch.c` — per‑stream branch helpers
  - `control.c` — simple HTTP control
  - `config.c` — config parsing helpers
  - `log.h`, `state.h` — macros and shared state

## Troubleshooting
- `/test` works but `/sN` returns 503:
  - Confirm logs show: `Linked demux src_N to UDP egress port 5000+N` and `RTSP mounted ... (udp-wrap H264 RTP @127.0.0.1:5000+N)`.
  - Ensure `pgie.txt` `batch-size` is >= expected concurrent sources. First run after bumping batch-size builds a new engine; allow ~1–2 minutes for b64.
- Port conflicts:
  - RTSP retries 8554..+9. Use the logged port in your ffplay URL.
  - DeepStream’s REST (9000/9010) is independent of RTSP; logs are informational.
- If using NVENC and the 8th/9th stream fails:
  - Confirm encoder init errors and consider staggering adds, lowering bitrate, or using software encoding as default.

## Branch Profiles
- `master` (C): production path, batch‑64 pre‑demux + x264 post‑demux, engine cached under `./models`. Control API on 8080.
- `c-b8-config` (C): batch‑8 variant to reduce inference memory footprint and ease NVENC pressure; `nvmultiurisrcbin port=9000` enabled. Still supports 64 streams (micro‑batching).
- `python-try` (Python): readable/dev server with Flask on `CONTROL_PORT` (default 8081). Mirrors C pipeline with NVENC tuning and staggered adds. On this host, NVENC sessions fail ~8–10; use for dev, not for 64‑stream scale.

## RTSP Wrapping Notes
- Use `udpsrc name=pay0` and carry payload in caps: `application/x-rtp, media=video, encoding-name=H264, clock-rate=90000, payload=96`.
- Do not set `pt` on `udpsrc` (not a valid property). `rtph264pay` can set `pt`, and caps should include `payload`.

## Scaling Considerations
- Ensure `max-batch-size` in `pipeline.txt` and `batch-size` in `pgie.txt` cover expected concurrent sources.
- Bandwidth planning: e.g., 64× streams at ~3 Mbps ≈ 192 Mbps aggregate.
- Encoder settings: keep `insert-sps-pps=1` and IDR/I‑frame intervals aligned to framerate; default bitrate ~3 Mbps @ 720p30.

## Style
- C: consistent naming, early returns, minimal globals, no dead paths.
- Use 80–100 char lines; whitespace around operators and commas.
- One config until stable: avoid options creep until `/sN` are reliable.
- Patches: keep changes surgical; avoid unrelated edits.
- Docs: concise, task‑oriented, and aligned to actual code.
