# Repository Structure

- Dockerfile — Builds the runtime image (DeepStream 8.0 base, compiles C server).
- build.sh — One‑shot Docker image build helper.
- run.sh — Runs the C server and passes env vars (`RTSP_PORT`, `BASE_UDP_PORT`, `PUBLIC_HOST`, `CTRL_PORT`, `HW_THRESHOLD`, `SW_MAX`, `MAX_STREAMS`).
  - Mounts `./models` on host to `/models` in the container to persist the TensorRT engine across runs.
- README.md — Architecture overview and usage.
- plan.md — Current implementation plan and next steps.
- STANDARDS.md — How to run/test; code cleanliness expectations.
- pipeline.txt — Pre‑demux pipeline (config‑driven): `nvmultiurisrcbin [uri-list=...] → nvinfer(pgie.txt) → nvstreamdemux name=demux`.
- pgie.txt — Primary GIE (nvinfer) configuration.
- src/main.c — Tiny entrypoint: parse env, sanity check, `app_setup()`, `app_loop()`, `app_teardown()`.
- src/app.{h,c} — App lifecycle and RTSP/pipeline setup:
  - `sanity_check_plugins()`, `decide_max_streams()` (hard limits),
  - `build_full_pipeline_and_server()` (reads `pipeline.txt` if present; mounts `/test`),
  - Bootstraps branches from `uri-list` count.
- src/branch.{h,c} — Per‑stream branch build and UDP egress; mounts RTSP endpoints.
  - Policy: first N NVENC, remaining SW encoders (`x264enc` → `avenc_h264` → `openh264enc`).
- src/control.{h,c} — Tiny HTTP control server (`/add_demo_stream`, `/status`).
- src/config.{h,c} — Configuration helpers.
- src/log.h, src/state.h — Logging macros and shared state.

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
- RTSP wrap uses `udpsrc` with H264 RTP caps and re‑payloads to `rtph264pay name=pay0` (gst-rtsp-server requires `pay0`).
- Pacing: sources flagged live (`live-source=1`), batched‑push‑timeout ~33 ms, and `udpsink sync=true` keep playback at realtime.
- HEALTHCHECK pings `/status` on `$CTRL_PORT`. Sanity checks ensure required plugins are present before running.
