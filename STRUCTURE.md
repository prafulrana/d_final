# Project Structure

```
d_final/
├── AGENTS.md                    # Architecture + workflow guide
├── STANDARDS.md                 # Coding/config standards
├── STRUCTURE.md                 # This file
├── start.sh                     # Launcher for both containers
├── config/
│   ├── s0_live.txt             # DeepStream config for live RTSP (in_s0 → s0)
│   ├── file_s3_s4.txt          # DeepStream config for file loops (s3, s4)
│   ├── config_infer_primary.txt # Shared inference config (TrafficCamNet/YOLO)
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
- **ds-s0**: Live RTSP input (`s0_live.txt`)
- **ds-files**: File loop tests (`file_s3_s4.txt`)

### config/s0_live.txt
- Source: RTSP from `in_s0` (camera via relay)
- Sink: Local RTSP server on localhost:8554
- Shared inference from `config_infer_primary.txt`

### config/file_s3_s4.txt
- Sources: Sample H.264 files (looping)
- Sinks: Local RTSP servers on localhost:8557 (s3), localhost:8558 (s4)
- Same inference config

### config/config_infer_primary.txt
- Model: TrafficCamNet ONNX (resnet18, 4 classes, FP16)
- Batch size: 1 (matches both s0 and s3/s4 streammux)
- Swappable for YOLO or other detectors

Nothing else is needed for the live demo.
