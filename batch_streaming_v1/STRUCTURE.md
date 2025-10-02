# Repository Structure

- Dockerfile — Builds the runtime image (DeepStream 8.0 base, compiles C server).
- build.sh — One‑shot Docker image build helper.
- run.sh — Runs the C server and passes env vars (`RTSP_PORT`, `BASE_UDP_PORT`, `PUBLIC_HOST`, `CTRL_PORT`, `HW_THRESHOLD`, `SW_MAX`, `MAX_STREAMS`).
  - Also mounts `./models` on host to `/models` in the container to persist the TensorRT engine across runs.
- README.md — Architecture overview and usage.
- plan.md — Current implementation plan and next steps.
- STANDARDS.md — How to run/test; code cleanliness expectations.
- (no pipeline.txt) — DeepStream stage is built in code: `nvmultiurisrcbin → nvinfer(pgie.txt) → nvstreamdemux name=demux`, and per‑stream overlays use `nvosd` (nvosdbin) after demux. Post‑demux encoding/RTSP is built by C.
- pgie.txt — Primary GIE (nvinfer) configuration.
- src/main.c — Tiny entrypoint calls a few small functions:
  - Reads config (args/env), does sanity checks, then `app_setup()`, `app_loop()`, `app_teardown()`.
- src/app.{h,c} — L1 app lifecycle and RTSP/pipeline setup:
  - `sanity_check_plugins()`, `decide_max_streams()` (hard limits),
  - `build_full_pipeline_and_server()` (loads pipeline.txt, mounts `/test`),
  - `app_setup()`, `app_loop()`, `app_teardown()`.
- src/branch.{h,c} — L2 per‑stream branch build:
  - `add_branch_and_mount()` calls small helpers to create and link elements, then mounts RTSP endpoint.
  - Per‑stream policy (hard limits): first 8 NVENC, remaining SW encoders (x264 → avenc → openh264)+CPU hop.
- src/control.{h,c} — Tiny HTTP control server:
  - `GET /add_demo_stream` adds a stream and triggers DeepStream REST add.
  - `GET /status` returns `{ max, streams: [...] }`.
- src/config.{h,c} — L2 configuration:
  - `AppConfig`, `parse_args()`, `cleanup_config()`, and `read_file_to_string()`.
- src/log.h — Logging macros for consistent output; src/state.h — shared state + per‑stream metadata.
- deepstream-8.0/ — Vendor assets and helper scripts (not modified by this app).

Branch matrix
- `master`
  - C production path; batch‑64; engine cached under `./models`.
  - Control API: `GET /add_demo_stream` on 8080.
  - RTSP wrap: udpsrc name=pay0 with latency=100.
- `c-b8-config`
  - Same as master, but batch‑8 (mux+PGIE) and `nvmultiurisrcbin port=9000` in `pipeline.txt` to ensure REST is enabled.
  - Use if GPU shows NVENC pressure with batch‑64; you can still run 64 streams.
- `python-try` (dev)
  - Python GI + Flask (`CONTROL_PORT` default 8081). Pipeline mirrors C, NVENC tuned, staggered adds, engine caching.
  - On this host, NVENC sessions fail ~8–10; use for readability/dev, not for 64‑stream scale.

Notes
- The Dockerfile includes `gstreamer1.0-plugins-ugly` and `gstreamer1.0-libav` so `x264enc`/`avenc_h264` are available.
- RTSP wrap uses `udpsrc name=pay0` and RTP caps to carry payload type. Do not set `pt` on `udpsrc`.
- A container HEALTHCHECK pings `/status` on `$CTRL_PORT`. Startup sanity checks ensure required plugins are present before running.
