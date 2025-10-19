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
1. Add new ONNX to `models/` directory
2. Edit `config/config_infer_primary.txt`:
   ```ini
   onnx-file=/models/your_model.onnx
   model-engine-file=/models/your_model.onnx_b1_gpu0_fp16.engine  # Optional, auto-generates if omitted
   num-detected-classes=<number>
   ```
3. Note batch-size requirements:
   - s0 Python script: batch-size=1 (line 45 in s0_rtsp.py)
   - s3/s4 config: batch-size=2 (in config/file_s3_s4.txt)
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
