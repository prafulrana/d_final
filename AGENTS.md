# Repository Guidelines

This repo is intentionally tiny. It exists only to drive the single-stream DeepStream demo you see at `http://<relay-ip>:8889/s0/`.

## What matters
- `config/rtsp_smoketest.txt` – DeepStream app config (TrafficCamNet, 30 FPS, single tile).
- `config/config_infer_primary.txt` – Detector configuration wired to the vendored ONNX + labels.
- `config/frpc.ini` – Sample `frpc` profile that matches the relay’s token/ports.
- `models/` – Only the TrafficCamNet ONNX (`resnet18_trafficcamnet_pruned.onnx`) and the 4-class label file (`labels.txt`).
- `relay/` – Terraform and startup script that bring up MediaMTX + FRPS on GCP with paths `in_s0` (publish) and `s0` (playback).

## Expectations for future edits
1. **Stay config-only.** Do not reintroduce C++, Python, or build systems unless the user explicitly requests it.
2. **Keep DeepStream references absolute.** Inside the container everything references `/config/...` and `/models/...`.
3. **30 FPS is sacred.** Retain `live-source=1`, `batch-size=1`, `batched-push-timeout=33333`, `iframeinterval=30`, `control-rate=1` unless told otherwise.
4. **Document behaviour.** If you adjust performance/format knobs, update these docs (and the commit message) so the next operator understands why.

## Quick workflow reminders
- Deploy relay: `cd relay && terraform init && terraform apply -var project_id=<gcp project>`.
- Start demo: `docker run -d --gpus all --network host -v "$(pwd)/config":/config -v "$(pwd)/models":/models nvcr.io/nvidia/deepstream:8.0-triton-multiarch deepstream-app -c /config/rtsp_smoketest.txt`.
- Optional tunnel: `frpc -c config/frpc.ini` when you need to forward RTSP through CG-NAT.
- Verify stream: open `http://<relay-ip>:8889/s0/` and confirm the single TrafficCamNet view is smooth with labels.

That’s it—stay lightweight, config-driven, and keep the stream clean.
