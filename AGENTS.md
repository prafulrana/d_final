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

## Notes for Agents
- Keep edits within the repo root; align with `STRUCTURE.md` and `STANDARDS.md`.
- Avoid new frameworks; prefer small, surgical changes to `app.c`, `branch.c`, `control.c`, `config.c`.

## Common Pitfalls & Checks
- Ensure remote RTSP host is reachable from the container host.
- Encoder props: `nvv4l2h264enc` varies by platform. Guard property sets with `g_object_class_find_property` (no hard-coded `maxperf-enable`).
- Startup order: create/mount branches before setting pipeline to `PLAYING` to avoid “data flow before segment” warnings.
- File inputs: in `pipeline.txt`, prefer `sync-inputs=true` and a larger `batched-push-timeout` for file URIs; align `pgie` batch-size/engine with `uri-list` or expect an engine rebuild.
