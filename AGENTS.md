# Repository Guidelines

This repo runs a 2-container DeepStream harness for testing TrafficCamNet inference on both live RTSP streams and file loops.

## Architecture

### Container 1: ds-s0 (Live RTSP)
- **Implementation**: Python script (`s0_rtsp.py`)
- **Source**: RTSP stream from `rtsp://34.14.140.30:8554/in_s0` (camera via MediaMTX relay)
- **Sink**: Local RTSP server on `localhost:8554/ds-test` (MediaMTX relay pulls from this)
- **Purpose**: Real-world streaming with TrafficCamNet inference at 30 FPS
- **Why Python**: Config-only approach causes segfaults; Python script handles nvurisrcbin dynamic pads properly

### Container 2: ds-files (File Loops)
- **Config**: `config/file_s3_s4.txt`
- **Source**: Sample H.264 files looping
- **Sinks**:
  - s3: Local RTSP server on `localhost:8557/ds-test`
  - s4: Local RTSP server on `localhost:8558/ds-test`
- **Purpose**: Deterministic testing/benchmarking

### Shared Pipeline
Both containers use the **same inference/OSD configuration**:
- **Inference**: `config/config_infer_primary.txt` (TrafficCamNet ONNX, batch-size=1, FP16)
- **OSD**: Bounding boxes + labels for Car/Person/Bicycle/RoadSign
- **Tracker**: (can be added to both configs)

## What Made s0 Smooth

The key was matching the architecture of the working file streams (s3/s4):
1. **Local RTSP server** instead of rtspclientsink publishing to remote MediaMTX
2. **MediaMTX pulls FROM localhost** as an RTSP client (relay config has `source: rtsp://localhost:8554/ds-test`)
3. **Consistent batch-size=1** everywhere (streammux + inference config + cached engine)
4. **nvurisrcbin** (type=4) handles RTSP connection/reconnection automatically
5. **NV12 format** from nvvideoconvert (encoder expects this)

## MediaMTX Relay Configuration

The MediaMTX relay at `34.14.140.30` needs this in `/etc/mediamtx/config.yml`:

```yaml
paths:
  in_s0:
    # Camera publishes here (source=publisher or runOnInit with ffmpeg)

  s0:
    source: rtsp://<this-machine-ip>:8554/ds-test
    sourceProtocol: tcp
    sourceOnDemand: no

  s3:
    source: rtsp://<this-machine-ip>:8557/ds-test
    sourceProtocol: tcp

  s4:
    source: rtsp://<this-machine-ip>:8558/ds-test
    sourceProtocol: tcp
```

Replace `<this-machine-ip>` with the IP where ds-s0/ds-files containers run.

## Expectations for Future Edits

1. **s0 uses Python, s3/s4 use config.** s0 requires Python script for smooth RTSP streaming. s3/s4 use vanilla `deepstream-app` with .txt configs.
2. **Shared inference config.** When swapping models (e.g., YOLO), update `config_infer_primary.txt` once and both containers pick it up.
3. **30 FPS is sacred.** Retain `batched-push-timeout=33333`, `iframeinterval=30`, `live-source=0` unless testing specific scenarios.
4. **Batch size consistency.** s0 Python script uses batch-size=1. s3/s4 use batch-size=2 for 2 file sources. Inference config must match.
5. **Document behavior.** Update this file when changing pipeline structure or performance knobs.

## Quick Workflow

### Start both containers
```bash
./start.sh
```

### Stop containers
```bash
docker stop ds-s0 ds-files
docker rm ds-s0 ds-files
```

### Check performance
```bash
docker logs ds-s0 | grep "**PERF"
docker logs ds-files | grep "**PERF"
```

### Verify streams
- s0 (live): `http://34.14.140.30:8889/s0/`
- s3 (file): `http://34.14.140.30:8889/s3/`
- s4 (file): `http://34.14.140.30:8889/s4/`

## Engine Caching (Manual)

After first startup, engines are built and saved in `models/`:
- `yolov8n_b1_gpu0_fp16.engine` (for s0, batch-size=1)
- `yolov8n_b2_gpu0_fp16.engine` (for s3/s4, batch-size=2)

To cache engines manually after build:
```bash
# Engines auto-save to models/ on first run
ls -lh models/*.engine

# Copy to backup if needed
cp models/yolov8n_b1_gpu0_fp16.engine models/yolov8n_b1_backup.engine
```

## Swapping Models (TrafficCamNet ↔ YOLOv8n ↔ YOLOWorld)

Three inference configs are included:
- `config_infer_primary.txt`: TrafficCamNet (4 classes: Car, Person, Bicycle, RoadSign)
- `config_infer_yolov8.txt`: YOLOv8n (80 COCO classes)
- `config_infer_yoloworld.txt`: YOLOWorld custom detector (80 COCO classes)

**To switch models**:
1. Edit `s0_rtsp.py` line 51: change `config_infer_*.txt`
2. Edit `config/file_s3_s4.txt` line 96: change `config-file=`
3. Delete old engines: `rm models/*.engine`
4. Restart: `./start.sh`

Both containers will rebuild engines for the new model.

That's it—lightweight, config-driven, same pipeline for live + files.
