# Standards

## Read DeepStream Samples First
- Align with DeepStream 8.0 patterns for muxing/demuxing and inference.
- Diverge only with documented rationale.

## How To Run (Direct Push)
- Requirements: Docker with NVIDIA runtime (`--gpus all`).
- Build: `./build.sh`
- Edit sources: update `pipeline.txt` `uri-list=...`.
- Run: `./run.sh` (pushes directly to remote RTSP as per `RTSP_PUSH_URL_TMPL`).
- Default targets: `rtsp://34.14.144.178:8554/s%u` with `MAX_STREAMS=2` (s0,s1).

## Relay IaC (GCP)
- The `relay/` folder contains Terraform to provision a MediaMTX relay VM on GCE (default zone `asia-south1-c`).
- Deploy: `cd relay && terraform init && terraform apply -var project_id=<YOUR_PROJECT>`.
- Outputs the `external_ip`; WebRTC at `http://<ip>:8889/s0/`.
- Update config: SSH to VM and edit `/etc/mediamtx/config.yml`, then `sudo docker restart mediamtx`.

- Pre‑demux defined in `pipeline.txt`:
  - `nvmultiurisrcbin [uri-list] → nvinfer (pgie.txt) → nvstreamdemux name=demux`.
- Post‑demux in C (per‑stream OSD/encode/h264parse → rtspclientsink [internal payloader]). Queue tuned: `leaky=0`, `max-size-time=200ms`.
- Encoder policy: first `HW_THRESHOLD` with NVENC; others software (`x264enc` → `avenc_h264` → `openh264enc`).

## Encoder Options and Limits
- NVENC sessions are finite; failures surface as encoder init errors. Plan for software fallback.
- x264 defaults: `zerolatency`, `ultrafast`, `bitrate=3000`, `key-int-max=30`, `bframes=0`, `threads≈half cores` (cap 4). Override `X264_THREADS`.

## Environment Variables
- `RTSP_PUSH_URL_TMPL` (default `rtsp://34.14.144.178:8554/s%u`)
- `MAX_STREAMS` (default `2`)
- `CTRL_PORT` (default `8080`)
- `HW_THRESHOLD` (default `8`), `SW_MAX` (default `56`), `X264_THREADS` (default `2`)

## Pacing & Engine
- Live sources: `nvmultiurisrcbin live-source=1`.
- Batch pacing: `batched-push-timeout=33000` (~30 fps).
- Engine cache under `./models`; delete to rebuild.

## Regression Guardrails
- Ensure the remote RTSP server is reachable from the container host.
- HW encoder props: do not set properties blindly on `nvv4l2h264enc`. Guard with `g_object_class_find_property` (e.g., avoid `maxperf-enable` on platforms that lack it).
- Start branches before PLAYING to reduce early segment/data-flow warnings.
- File inputs: in `pipeline.txt` use `sync-inputs=true` and a higher `batched-push-timeout` (e.g., `100000`). Ensure `pgie` batch-size (and engine) is compatible with the number of URIs, or allow the engine to rebuild.

## Troubleshooting
- Check logs for `Linked demux src_N to rtspclientsink → ...` and any encoder errors; verify remote server path exists.

## Style
- C style: clear naming, early returns, minimal globals. Keep changes surgical; avoid unrelated edits. Update docs with behavior changes.
