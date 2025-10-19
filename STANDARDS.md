# Standards and Workflow

## DeepStream Config Changes

### Editing Configs
- **Inference** (shared): Edit `config/config_infer_primary.txt` to change model/detection settings
- **OSD/Overlay** (shared):
  1. Edit `config/config_osd.txt` to change bounding box colors, text size, fonts
  2. Run `./update_osd.sh` to sync settings to s0_rtsp.py and file_s3_s4.txt
- **Pipeline structure**: Edit `s0_rtsp.py` or `config/file_s3_s4.txt` directly only for pipeline changes
- Restart containers after changes:
  ```bash
  ./start.sh
  ```

### Changing Inference Model

**Quick Swap (TrafficCamNet ↔ YOLOWorld)**:
```bash
# Option 1: TrafficCamNet (default, 4 classes)
sed -i 's|config_infer_yoloworld.txt|config_infer_primary.txt|' s0_rtsp.py config/file_s3_s4.txt

# Option 2: YOLOWorld (custom detector, network-type=100)
sed -i 's|config_infer_primary.txt|config_infer_yoloworld.txt|' s0_rtsp.py config/file_s3_s4.txt

# Apply changes
rm models/*.engine  # Clear cached engines
./start.sh
```

**Custom Model**:
1. Add new ONNX to `models/` directory
2. Create `config/config_infer_yourmodel.txt`:
   ```ini
   onnx-file=/models/your_model.onnx
   num-detected-classes=<number>
   batch-size=1  # or 2 for s3/s4 dual sources
   ```
3. Point configs to new file: `sed -i 's|config_infer_primary.txt|config_infer_yourmodel.txt|' s0_rtsp.py config/file_s3_s4.txt`
4. Delete old engine files: `rm models/*.engine`
5. Restart: `./start.sh`

### Performance Validation
Check FPS in container logs:
```bash
# Check s0 (live RTSP)
docker logs ds-s0 | grep "**PERF" | tail

# Check s3/s4 (file loops)
docker logs ds-files | grep "**PERF" | tail
```

You should see ~30 FPS steady-state.

### Debugging Streams
View streams in browser:
- s0 (live): http://34.14.140.30:8889/s0/
- s3 (file): http://34.14.140.30:8889/s3/
- s4 (file): http://34.14.140.30:8889/s4/

Check MediaMTX relay logs:
```bash
gcloud compute ssh mediamtx-relay --zone=asia-south1-c --command="docker logs mediamtx --tail 50"
```

## Relay Edits

### Modifying MediaMTX Config
The relay at `34.14.140.30` needs to pull from your DeepStream containers:

```yaml
paths:
  in_s0:
    # Camera publishes here

  s0:
    source: rtsp://<your-machine-ip>:8554/ds-test
    sourceProtocol: tcp
    sourceOnDemand: no

  s3:
    source: rtsp://<your-machine-ip>:8557/ds-test
    sourceProtocol: tcp

  s4:
    source: rtsp://<your-machine-ip>:8558/ds-test
    sourceProtocol: tcp
```

### Terraform Changes
- Modify `relay/main.tf` or `relay/scripts/startup.sh`
- Run `terraform plan` before `terraform apply`
- Destroy/recreate VM after startup script changes

## FRP Usage (NAT Traversal)

If your DeepStream machine is behind NAT:
```bash
# Edit config/frpc.ini with your relay's token
frpc -c config/frpc.ini

# Verify tunnel
docker logs frpc --tail 20
```

## Git Practices

### Committing Changes
- Large binaries (ONNX/engine files) tracked via Git LFS
- Commit messages: `scope: imperative summary`
  - Examples:
    - `config: switch to YOLOv8 detector`
    - `relay: add path for s5 stream`
    - `docs: update YOLO model swap instructions`

### What to Commit
- **Do commit**: Config files, docs, scripts
- **Do NOT commit**: TensorRT .engine files (they're machine-specific, auto-generated)

## Key Constraints

1. **30 FPS is sacred**: Keep `batched-push-timeout=33333`, `iframeinterval=30`
2. **Batch-size awareness**: s0 uses batch-size=1, s3/s4 use batch-size=2
3. **live-source=0**: Always use this (even for RTSP) for consistent frame pacing
4. **Python for s0**: Config-only approach causes segfaults; use s0_rtsp.py

Keep it simple: Python for s0, config for s3/s4, shared inference pipeline.
