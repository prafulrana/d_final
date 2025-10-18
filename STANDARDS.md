# Standards and Workflow

## DeepStream config changes
- Edit files under `config/`.
- Restart the demo container after changes:
  ```bash
  docker rm -f ds-demo
  docker run -d --gpus all --name ds-demo --network host \
    -v "$(pwd)/config":/config \
    -v "$(pwd)/models":/models \
    nvcr.io/nvidia/deepstream:8.0-triton-multiarch \
    deepstream-app -c /config/rtsp_smoketest.txt
  ```
- Validate with `docker logs ds-demo | grep "**PERF" | tail` – you should see ~30 FPS steady-state.

## Relay edits
- Modify Terraform or `scripts/startup.sh` inside `relay/`.
- Run `terraform plan` before `terraform apply`.
- After startup script changes, rebuild the VM (destroy/apply) or manually rewrite `/etc/mediamtx/config.yml` and restart `mediamtx`.
- Confirm MediaMTX logs no longer show `request timed out` once the pipeline is live.

## FRP usage
- `config/frpc.ini` is the canonical template.
- Launch `frpc -c config/frpc.ini` and ensure the log prints `start proxy success`.

## Git practices
- Large binaries (ONNX/labels) stay under `models/` and are already tracked via Git LFS.
- Commit messages follow `scope: imperative summary` (e.g., `config: retune source30 to 30fps`).
- Update these docs when workflows or assumptions change.

Keep it simple: configs only, relay provisioning, and a single DeepStream container.
