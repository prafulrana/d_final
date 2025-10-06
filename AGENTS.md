# Repository Guidelines

This repo hosts a C-based GStreamer/DeepStream RTSP server in `batch_streaming_v1/`. The pre‑demux pipeline is defined via `pipeline.txt` (config-driven), while post‑demux encoding and RTSP wrapping are built in C.

## Project Structure & Modules
- `batch_streaming_v1/pipeline.txt` — Pre‑demux: `nvmultiurisrcbin → nvinfer (pgie.txt) → nvstreamdemux name=demux` with `uri-list=...`.
- `batch_streaming_v1/src/` — C sources: `main.c` (entry), `app.c` (lifecycle + pipeline/RTSP), `branch.c` (per‑stream), `control.c` (HTTP), `config.c` (args/env), headers.
- `Dockerfile`, `build.sh`, `run.sh`, `sanity.sh`, `README.md`, `STANDARDS.md`, `STRUCTURE.md`, `pgie.txt`.
- Engine cache persists under `./models` (mounted by `run.sh`).

## Build, Test, Run
- Build: `cd batch_streaming_v1 && ./build.sh` (includes `sanity.sh`).
- Run: `./run.sh` (envs: `RTSP_PORT`, `PUBLIC_HOST`, `CTRL_PORT`, `BASE_UDP_PORT`).
- Control API: `curl http://localhost:8080/status` and `curl http://localhost:8080/add_demo_stream` (optional adds beyond `uri-list`).
- Play: `ffplay -rtsp_transport tcp rtsp://<host>:8554/s0` (also `/s1`, ... if present).

## Coding Style & Conventions
- C with GLib/GStreamer. Indent 2 spaces; 80–100 cols. File names `lower_snake_case.c/.h`; symbols `snake_case`; globals prefixed `g_`.
- Use logging macros in `log.h` (`LOG_ERR/WRN/INF`). Keep functions short with early returns.
- Do not expand pre‑demux in code—prefer `pipeline.txt`. Post‑demux (branch/encode/RTSP) stays in C.

## Testing Guidelines
- Sanity inside container: `docker run --rm -i batch_streaming:latest bash -s < sanity.sh`.
- End‑to‑end: edit `pipeline.txt` URIs, `./run.sh`, confirm `/status`, then `ffplay` returned endpoints.
- Include logs and minimal repro for any pipeline or encoder changes.

## Commits & Pull Requests
- Commit style: `Scope: imperative summary` (e.g., `Server: bootstrap branches from uri-list`).
- PRs must include: change rationale, test steps, configs/envs, and doc updates when behavior changes (`STANDARDS.md`, `STRUCTURE.md`).

## Notes for Agents
- Keep edits within `batch_streaming_v1/`; align with `STRUCTURE.md` and `STANDARDS.md`.
- Avoid new frameworks; prefer small, surgical changes to `app.c`, `branch.c`, `control.c`, `config.c`.
