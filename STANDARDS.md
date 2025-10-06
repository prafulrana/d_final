# Standards

## Read DeepStream Samples First
- Align with DeepStream 8.0 patterns:
  - `sources/apps/apps-common/src/deepstream_sink_bin.c` (UDP→RTSP: depay→pay name=pay0).
  - `sources/apps/sample_apps/deepstream-app` (demux/sink wiring, config parsing).
- Diverge only with documented rationale.

## How To Run (Pipeline File)
- Requirements: Docker with NVIDIA runtime (`--gpus all`).
- Build: `./build.sh`
- Edit sources: update `pipeline.txt` `uri-list=...`.
- Run: `./run.sh` (starts `/test` and bootstraps `/sN` for each URI).
- Add more (optional): `curl http://localhost:8080/add_demo_stream`.
- Play: `ffplay -rtsp_transport tcp rtsp://<host>:8554/s0`

## Configuration Strategy
- Pre‑demux defined in `pipeline.txt`:
  - `nvmultiurisrcbin [uri-list] → nvinfer (pgie.txt) → nvstreamdemux name=demux`.
- Post‑demux in C (per‑stream OSD/encode/RTP/UDP→RTSP). Queue tuned: `leaky=0`, `max-size-time=200ms`.
- Encoder policy: first `HW_THRESHOLD` with NVENC; others software (`x264enc` → `avenc_h264` → `openh264enc`). RTP/UDP wrapped to RTSP with `rtph264pay name=pay0`.

## Encoder Options and Limits
- NVENC sessions are finite; failures surface as encoder init errors. Plan for software fallback.
- x264 defaults: `zerolatency`, `ultrafast`, `bitrate=3000`, `key-int-max=30`, `bframes=0`, `threads≈half cores` (cap 4). Override `X264_THREADS`.

## Environment Variables
- `RTSP_PORT` (8554), `BASE_UDP_PORT` (5000), `PUBLIC_HOST` (127.0.0.1), `CTRL_PORT` (8080), `HW_THRESHOLD` (8), `SW_MAX` (56), `MAX_STREAMS` (HW+SW), `X264_THREADS` (2).

## Pacing & Engine
- Live sources: `nvmultiurisrcbin live-source=1`.
- Batch pacing: `batched-push-timeout=33000` (~30 fps).
- Output pacing: `udpsink sync=true` honors timestamps.
- Engine cache under `./models`; delete to rebuild.

## Troubleshooting
- `/test` works but `/sN` fails: check logs for `Linked demux src_N ...` and RTSP mount lines; verify `pgie.txt` batch-size ≥ number of URIs.
- Ports: RTSP retries 8554..+9; use the logged port. Ensure `BASE_UDP_PORT` range is free.

## Style
- C style: clear naming, early returns, minimal globals. Keep changes surgical; avoid unrelated edits. Update docs with behavior changes.
