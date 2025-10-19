# Standards and Workflow

## DeepStream Config Changes

### Editing Configs
- Edit `.txt` files under `config/` directory
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
   batch-size=1  # Keep at 1 for single-stream
   ```
3. Update batch-size in **both** `config/s0_live.txt` and `config/file_s3_s4.txt` if needed:
   ```ini
   [streammux]
   batch-size=1  # Must match config_infer_primary.txt
   ```
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
2. **Batch-size consistency**: streammux batch-size MUST match inference config batch-size
3. **live-source=0**: Always use this (even for RTSP) for consistent frame pacing
4. **Stay config-only**: Use vanilla deepstream-app unless custom probes/trackers are explicitly needed

Keep it simple: configs only, 2 containers, shared inference pipeline.
