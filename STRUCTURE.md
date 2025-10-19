# Project Structure

```
d_final/
├── AGENTS.md                    # Architecture + workflow guide
├── STANDARDS.md                 # Coding/config standards
├── STRUCTURE.md                 # This file
├── start.sh                     # Launcher for both containers
├── update_osd.sh                # Sync OSD settings from config_osd.txt to both pipelines
├── s0_rtsp.py                   # Python script for s0 (live RTSP)
├── config/
│   ├── file_s3_s4.txt          # DeepStream config for file loops (s3, s4)
│   ├── config_infer_primary.txt # Shared inference config (TrafficCamNet/YOLO)
│   ├── config_osd.txt          # Shared OSD config (bounding boxes, text)
│   └── frpc.ini                # FRP client template for NAT traversal
├── models/
│   ├── labels.txt              # Class labels (Car, Person, Bicycle, RoadSign)
│   ├── resnet18_trafficcamnet_pruned.onnx  # TrafficCamNet detector (Git LFS)
│   └── *.engine                # TensorRT engine cache (auto-generated)
└── relay/
    ├── main.tf                 # Terraform for MediaMTX relay VM
    ├── scripts/startup.sh      # VM startup script (MediaMTX + FRPS)
    ├── test.sh                 # Relay port probe helper
    ├── variables.tf            # Terraform variables
    └── README.md               # Relay deployment docs
```

## Key Files

### start.sh
Launches 2 containers:
- **ds-s0**: Live RTSP input (Python script `s0_rtsp.py`)
- **ds-files**: File loop tests (`file_s3_s4.txt`)

### s0_rtsp.py
- Python script for s0 (config-only approach causes segfaults)
- Source: RTSP from `in_s0` (camera via relay)
- Sink: Local RTSP server on localhost:8554
- Shared inference from `/config/config_infer_primary.txt`

### config/file_s3_s4.txt
- Sources: Sample H.264 files (looping)
- Sinks: Local RTSP servers on localhost:8557 (s3), localhost:8558 (s4)
- Same inference config

### config/config_infer_primary.txt
- Model: TrafficCamNet ONNX (resnet18, 4 classes, FP16)
- Used by both s0 Python script and s3/s4 config files
- Swappable for YOLO or other detectors

### config/config_osd.txt
- Shared OSD (overlay) settings source of truth
- Bounding box appearance: border-width, colors
- Text appearance: text-size, text-color, font
- After editing, run `./update_osd.sh` to sync to both s0 and s3/s4

### update_osd.sh
- Parses `config/config_osd.txt`
- Updates OSD properties in `s0_rtsp.py` (Python set_property calls)
- Updates `[osd]` section in `config/file_s3_s4.txt`
- Ensures identical OSD settings across all streams

Note: s0 uses batch-size=1 (single stream), s3/s4 use batch-size=2 (dual file sources).
