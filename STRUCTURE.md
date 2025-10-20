# Project Structure

```
d_final/
├── README.md                    # Quick start guide
├── AGENTS.md                    # Architecture + operational guide
├── STANDARDS.md                 # Coding/config standards
├── STRUCTURE.md                 # This file
├── system.sh                    # Manage local (start/stop/restart/status)
├── relay.sh                     # Manage remote relay (restart/status)
├── check.sh                     # Health check validation
├── debug.sh                     # Debugging diagnostics
├── start.sh                     # Start containers only
├── stop.sh                      # Stop containers only
├── build.sh                     # Docker image builder
├── frpc-{start,stop,restart,status}.sh  # frpc tunnel management
├── live_stream.c                # Parameterized C binary (takes stream ID 0/1/2)
├── Dockerfile.s1                # Builds ds-s1:latest with live_stream binary
├── libnvdsinfer_custom_impl_Yolo.so  # YOLOv8 custom parser library
├── config/
│   ├── config_infer_yolov8.txt # YOLOv8n inference config
│   └── frpc.ini                # FRP client config (CONTAINS RELAY IP)
├── models/
│   ├── yolov8n.onnx            # YOLOv8n model
│   ├── coco_labels.txt         # COCO class labels
│   └── *.engine                # TensorRT engine cache (auto-generated)
├── publisher/                   # Test video publisher (isolated)
│   ├── Dockerfile              # Publisher image build
│   ├── loop_stream.sh          # Publish script (loops video to in_s2)
│   ├── start.sh                # Publisher launcher
│   └── bowling_bottom_right.mp4  # Portrait test video
└── relay/
    ├── main.tf                 # Terraform for MediaMTX relay VM
    ├── scripts/startup.sh      # VM startup script (MediaMTX + frps server)
    ├── variables.tf            # Terraform variables
    ├── test.sh                 # End-to-end relay test
    └── README.md               # Relay deployment docs
```

## Key Files

### Live Stream (Parameterized C Binary)
- **live_stream.c**: Single binary, takes stream ID (0, 1, 2) as argv[1]
  - Stream ID 0 → pulls `in_s0`, serves on `8554`, UDP port `5400`
  - Stream ID 1 → pulls `in_s1`, serves on `8555`, UDP port `5401`
  - Stream ID 2 → pulls `in_s2`, serves on `8556`, UDP port `5402`
  - All use YOLOv8n inference (`config/config_infer_yolov8.txt`)
  - **CONTAINS RELAY IP** (line 97: derived from `snprintf`)

### Docker Images
- **Dockerfile.s1**: Builds single binary `live_stream` into one image
- **publisher/Dockerfile**: Builds test video publisher

### Scripts
- **start.sh**: Starts all containers (ds-s0, ds-s1, ds-s2, publisher)
- **stop.sh**: Stops all containers
- **build.sh**: Rebuilds ds-s1:latest and publisher:latest images
- **publisher/start.sh**: Alternative standalone publisher start (use main start.sh instead)
- **publisher/loop_stream.sh**: Publishes video to relay's in_s2

### Configuration
- **config/config_infer_yolov8.txt**: YOLOv8n detector (80 COCO classes)
- **config/config_osd.txt**: Bounding box colors, text size
- **config/frpc.ini**: FRP client settings (**CONTAINS RELAY IP AND TOKEN**)

### Relay Infrastructure
- **relay/main.tf**: Terraform config for GCP VM
- **relay/scripts/startup.sh**: Installs Docker, MediaMTX, frps
  - **IMMUTABLE**: Changes require `terraform destroy` + `terraform apply`
- **relay/README.md**: Deployment and troubleshooting guide

## Files Containing Relay IP (Update After IP Change)

1. `live_stream.c` (line 97: `snprintf(input_uri, ...)`)
2. `config/frpc.ini` (lines 2 and 4)
3. `publisher/loop_stream.sh` (line 7: `rtspclientsink location`)

## Architecture Summary

### Current Setup
- **3 DeepStream containers**: ds-s0, ds-s1, ds-s2
- **All pull from relay**: `rtsp://34.47.221.242:8554/in_s{0,1,2}`
- **All serve locally**: `rtsp://localhost:855{4,5,6}/ds-test`
- **frpc tunnels**: localhost:855X → relay:950X
- **Relay outputs**: http://34.47.221.242:8889/s{0,1,2}/

### Data Flow
```
Camera → in_s0 (relay) → ds-s0 (YOLOv8) → localhost:8554 → frpc → relay:9500 → s0 (WebRTC)
Camera → in_s1 (relay) → ds-s1 (YOLOv8) → localhost:8555 → frpc → relay:9501 → s1 (WebRTC)
Camera → in_s2 (relay) → ds-s2 (YOLOv8) → localhost:8556 → frpc → relay:9502 → s2 (WebRTC)
```

### Why This Architecture?
1. **Relay is public**: Acts as STUN server for WebRTC, accessible from anywhere
2. **DeepStream is local**: GPU machine behind NAT, can't accept incoming connections
3. **frp tunnels**: Bridge the gap - DeepStream serves RTSP locally, frpc tunnels to relay
4. **Relay pulls**: MediaMTX on relay pulls from tunneled RTSP servers

## Typical Workflows

### Start Everything
```bash
./build.sh    # Only if live_stream.c changed
./start.sh    # Start all: ds-s0, ds-s1, ds-s2, publisher
```

### Stop Everything
```bash
./stop.sh     # Stop all containers
```

### After Relay IP Change
```bash
# 1. Update live_stream.c line 97
# 2. Update config/frpc.ini lines 2, 4
# 3. Update publisher/loop_stream.sh line 7
./build.sh
pkill frpc && nohup frpc -c config/frpc.ini > /var/log/frpc.log 2>&1 &
./start.sh
```

### Change Inference Model
```bash
# Edit live_stream.c line 165 to use different config_infer_*.txt
sed -i 's|config_infer_yolov8.txt|config_infer_new.txt|' live_stream.c
./build.sh
rm models/*.engine
./start.sh
```

### Rebuild TensorRT Engine
```bash
rm models/*.engine    # Clear cached engines
./start.sh            # Let first container build, then restart others
```

## What NOT to Do

1. **Don't manually edit /etc/mediamtx/config.yml on relay**: Use `relay/scripts/startup.sh` + Terraform
2. **Don't skip rebuilding after .c changes**: Docker uses cached binary, not your edits
3. **Don't forget to restart frpc after config changes**: Old process keeps old IP/token
4. **Don't modify working streams when debugging**: If s0 works but s2 doesn't, leave s0 alone
5. **Don't start all 3 containers simultaneously on first engine build**: Causes race condition/deadlock
