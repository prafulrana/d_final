# Repository Structure

- Dockerfile — Builds the runtime image (DeepStream 8.0 base), binary `drishti`.
- build.sh — One‑shot Docker image build helper.
- run.sh — Launches a single sender container (`drishti`). Env: `RTSP_PUSH_URL_TMPL`, `CTRL_PORT`, `HW_THRESHOLD`, `SW_MAX`, `MAX_STREAMS`.
  - Defaults: target `rtsp://34.14.144.178:8554/s%u`, `MAX_STREAMS=2` (s0,s1).
  - Mounts `./models` on host to `/models` in the container to persist the TensorRT engine across runs.
- STANDARDS.md — How to run/test; code cleanliness expectations.
- pipeline.txt — Pre‑demux pipeline (config‑driven): `nvmultiurisrcbin [uri-list=...] → nvinfer(pgie.txt) → nvstreamdemux name=demux`.
- pgie.txt — Primary GIE (nvinfer) configuration.
- relay/ — Terraform IaC to deploy the MediaMTX relay VM on GCP:
  - `main.tf`, `variables.tf` — GCE instance + firewall (default zone `asia-south1-c`).
  - `scripts/startup.sh` — Installs Docker, writes `/etc/mediamtx/config.yml`, runs MediaMTX.
  - `README.md` — How to deploy, operate, and monitor the relay.
- src/main.c — Tiny entrypoint: parse env, sanity check, `app_setup()`, `app_loop()`, `app_teardown()`.
- src/app.{h,c} — App lifecycle and pipeline setup:
  - `sanity_check_plugins()`, `decide_max_streams()` (hard limits),
  - `build_pipeline()` (reads `pipeline.txt` if present; direct RTSP push),
  - Bootstraps branches from `uri-list` count.
- src/branch.{h,c} — Per‑stream branch build and egress; pushes RTSP directly via `rtspclientsink`.
  - Policy: first N NVENC, remaining SW encoders (`x264enc` → `avenc_h264` → `openh264enc`).
  - Optional: when `RTSP_PUSH_URL_TMPL` is set, branches publish directly to the remote RTSP via `rtspclientsink` instead of local RTSP.
- src/control.{h,c} — Tiny HTTP control server (`/add_demo_stream`, `/status`).
- src/config.{h,c} — Configuration helpers.
- src/log.h, src/state.h — Logging macros and shared state.

Branching
- Single mainline implementation; engine cache under `./models`.
- Control API: `GET /add_demo_stream` on 8080.
- Direct RTSP push via `rtspclientsink`.

Notes
- The Dockerfile includes `gstreamer1.0-plugins-ugly` and `gstreamer1.0-libav` so `x264enc`/`avenc_h264` are available.
- Pacing: sources flagged live (`live-source=1`) and `rtspclientsink` handles RTSP RECORD.
- HEALTHCHECK pings `/status` on `$CTRL_PORT`. Sanity checks ensure required plugins are present before running.

Operational notes (avoid regressions)
- Ensure the remote RTSP server is reachable from the container host.
- For file URIs in `pipeline.txt`, prefer `sync-inputs=true` and a higher `batched-push-timeout` (e.g., `100000`).
- Prepare and link branches before setting the pipeline to `PLAYING` to reduce early data-flow warnings.
- Guard platform-specific encoder properties on `nvv4l2h264enc` (check with `g_object_class_find_property`).
