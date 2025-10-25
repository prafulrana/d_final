# Project Structure

```
d_final/
├── README.md                    # Quick start guide
├── AGENTS.md                    # Architecture + operational guide
├── STANDARDS.md                 # Coding/config standards
├── STRUCTURE.md                 # This file
│
├── system                       # Full system manager (frpc + DeepStream)
├── ds                          # DeepStream containers only (start/stop/status)
├── relay                       # Remote relay manager (restart/status)
├── build.sh                    # Docker image builder
│
├── live_stream.c               # DRY C binary (takes stream ID 0-3, orientation, config)
├── Dockerfile.s1               # Builds ds-s1:latest with live_stream binary
├── libnvdsinfer_custom_impl_Yolo.so  # YOLOv8 custom parser library
│
├── .scripts/                   # Hidden utilities
│   ├── cache_engine.sh        # TensorRT engine cache manager
│   ├── check.sh               # Health check validation
│   ├── debug.sh               # Debugging diagnostics
│   └── frpc/
│       └── frpc.ini           # FRP client config (CONTAINS RELAY IP + TOKEN)
│
├── config/
│   ├── config_infer_yolov8.txt    # YOLOv8n COCO (s0-s2, 80 classes)
│   └── config_infer_bowling.txt   # YOLOv8n bowling (s3, 1 class)
│
├── models/
│   ├── yolov8n.onnx           # YOLOv8n COCO model (s0-s2)
│   ├── bowling.onnx           # YOLOv8n bowling model (s3)
│   ├── coco_labels.txt        # COCO class labels (80 classes)
│   ├── bowling_labels.txt     # Bowling class label (1 class)
│   └── *.engine               # TensorRT engine cache (auto-generated)
│
├── training/                  # Training artifacts (gitignored)
│   ├── scripts/
│   │   └── export_deepstream.sh  # DeepStream-Yolo ONNX export wrapper
│   ├── DeepStream-Yolo/       # Custom ONNX export scripts (cloned repo)
│   └── bowling/               # Dataset + training runs
│
└── relay_infra/               # Terraform infrastructure for relay
    ├── main.tf                # MediaMTX relay VM
    ├── scripts/startup.sh     # VM startup script (MediaMTX + frps server)
    └── variables.tf           # Terraform variables
```

## Training Directory

```
training/
├── train_bowling.py                        # Training script for bowling models
├── labels.txt                              # Class label (bowling-ball)
│
├── Bowling-Pin-Detection--4/               # Roboflow dataset (downloaded)
│   ├── train/images/                       # 868 training images
│   ├── valid/images/                       # 154 validation images
│   ├── test/images/                        # Test images
│   ├── data.yaml                           # Dataset config (1 class: bowling-ball)
│   └── README.roboflow.txt                 # Dataset info
│
├── bowling_real_dataset/                   # Custom labeled dataset (experimental)
│   └── ...auto-labeled frames
│
└── roboflow_bowling/                       # Training run outputs
    ├── bowling_1280_m_b8/                  # YOLOv8m @ 1280x1280, batch=8 ✓ PRODUCTION
    │   ├── weights/
    │   │   ├── best.pt                     # Epoch 70, 92.7% precision, 88.9% recall
    │   │   ├── best.onnx                   # Exported with dynamic=False
    │   │   └── last.pt                     # Final epoch checkpoint
    │   ├── results.png                     # Training curves
    │   └── confusion_matrix.png            # Validation metrics
    │
    └── bowling_1280_m/                     # Earlier run (batch=16, OOM issues)
        └── weights/
```

**Key Points:**
- `Bowling-Pin-Detection--4/` is gitignored (can re-download from Roboflow)
- Training outputs (`roboflow_bowling/`) are gitignored except metadata
- Production models copied to `/root/d_final/models/` and committed
- Use Ultralytics export with `dynamic=False` for DeepStream compatibility

## Key Files

### Management Scripts (Root Level)
- **system**: Full system management (start/stop/restart/status)
  - Manages frpc tunnel + DeepStream containers together
  - Use `./system start` to start frpc and all DS containers
  - Use `./system status` to check frpc, DS containers, and relay health
- **ds**: DeepStream containers only (build/start/stop/restart/status)
  - Use `./ds start` to start containers without touching frpc
- **relay**: Remote relay management (restart/status)
  - SSHs to relay VM and manages frps + mediamtx containers
  - Use `./relay status` to check relay health

### Live Stream (Parameterized C Binary)
- **live_stream.c**: Single binary, takes stream ID (0, 1, 2) as argv[1]
  - Stream ID 0 → pulls `in_s0`, serves on `8554`, UDP port `5400`
  - Stream ID 1 → pulls `in_s1`, serves on `8555`, UDP port `5401`
  - Stream ID 2 → pulls `in_s2`, serves on `8556`, UDP port `5402`
  - All use YOLOv8n inference (`config/config_infer_yolov8.txt`)
  - **CONTAINS RELAY IP** (line 96: `snprintf(input_uri, ...)`)

### Docker Images
- **Dockerfile.s1**: Builds single binary `live_stream` into one image (ds-s1:latest)

  - Automatically stops old container and starts new one

### Configuration Files
- **config/config_infer_yolov8.txt**: YOLOv8n detector (80 COCO classes)
- **.scripts/frpc/frpc.ini**: FRP client settings (**CONTAINS RELAY IP AND TOKEN**)

### Utilities (Hidden in .scripts/)
- **.scripts/cache_engine.sh**: TensorRT engine cache manager (copy/list/clean)
- **.scripts/check.sh**: Health check validation (used by `./system status`)
- **.scripts/debug.sh**: Debugging diagnostics (check container logs, RTSP servers, etc.)

### Relay Infrastructure
- **relay/main.tf**: Terraform config for GCP VM
- **relay/scripts/startup.sh**: Installs Docker, MediaMTX, frps
  - **IMMUTABLE**: Changes require `terraform destroy` + `terraform apply`
- **relay/README.md**: Deployment and troubleshooting guide

## Files Containing Relay IP (Update After IP Change)

1. `live_stream.c` (line 97: `snprintf(input_uri, ...)`)
2. `frpc/frpc.ini` (lines 2 and 4)

## Architecture Summary

### Current Setup
- **4 DeepStream containers**: ds-s0, ds-s1, ds-s2, ds-s3
- **All pull from relay**: `rtsp://34.47.221.242:8554/in_s{0,1,2,3}`
- **All serve locally**: `rtsp://localhost:855{4,5,6,7}/ds-test`
- **frpc tunnels**: localhost:855X → relay:950X
- **Relay outputs**: http://34.47.221.242:8889/s{0,1,2,3}/

### Data Flow
```
Camera → in_s0 (relay) → ds-s0 (COCO YOLOv8) → localhost:8554 → frpc → relay:9500 → s0 (WebRTC)
Camera → in_s1 (relay) → ds-s1 (COCO YOLOv8) → localhost:8555 → frpc → relay:9501 → s1 (WebRTC)
Camera → in_s2 (relay) → ds-s2 (COCO YOLOv8) → localhost:8556 → frpc → relay:9502 → s2 (WebRTC)
Bowling → in_s3 (relay) → ds-s3 (Bowling YOLOv8) → localhost:8557 → frpc → relay:9503 → s3 (WebRTC)
```

### Why This Architecture?
1. **Relay is public**: Acts as STUN server for WebRTC, accessible from anywhere
2. **DeepStream is local**: GPU machine behind NAT, can't accept incoming connections
3. **frp tunnels**: Bridge the gap - DeepStream serves RTSP locally, frpc tunnels to relay
4. **Relay pulls**: MediaMTX on relay pulls from tunneled RTSP servers

## Typical Workflows

### Start Everything
```bash
./build.sh         # Only if live_stream.c changed
```

### Stop Everything
```bash
./system stop      # Stop all containers + frpc
```

### DeepStream Only (without frpc)
```bash
./ds stop          # Stop all containers
./ds status        # Check container status
```

```bash
```

### Check System Health
```bash
./system status    # Full health check (frpc + DS + relay)
./ds status        # Just container status
./relay status     # Just relay status
```

### After Relay IP Change
```bash
# 1. Update live_stream.c line 96
# 2. Update .scripts/frpc/frpc.ini lines 2, 4
./build.sh
./system restart   # Restart frpc + containers with new config
```

### Change Inference Model
```bash
# Edit live_stream.c line 150 to use different config_infer_*.txt
sed -i 's|config_infer_yolov8.txt|config_infer_new.txt|' live_stream.c
./build.sh
rm models/*.engine
./ds restart
```

### Manage TensorRT Engine Cache
```bash
# Copy engines from containers to persistent cache (after first build)
./.scripts/cache_engine.sh copy

# Verify configs point to cached engines
./.scripts/cache_engine.sh verify

# List engines in containers and /models/
./.scripts/cache_engine.sh list

# Force rebuild (rarely needed - only if engine corrupted)
./.scripts/cache_engine.sh clean
./ds restart
./.scripts/cache_engine.sh copy  # Copy rebuilt engines back to cache
```

## What NOT to Do

1. **Don't manually edit /etc/mediamtx/config.yml on relay**: Use `relay_infra/scripts/startup.sh` + Terraform
2. **Don't skip rebuilding after .c changes**: Docker uses cached binary, not your edits
3. **Don't forget to restart system after config changes**: Use `./system restart` to apply new frpc config
4. **Don't modify working streams when debugging**: If s0 works but s2 doesn't, leave s0 alone
