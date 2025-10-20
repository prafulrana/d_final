# DeepStream RTSP YOLOv8 Pipeline

3 live RTSP streams with YOLOv8n object detection, tunneled to public relay via frp.

## Quick Start

```bash
./system.sh start     # Start everything (frpc + containers)
./system.sh status    # Check health
./system.sh stop      # Stop everything
```

## Architecture

```
Cameras → Relay (in_s0/1/2) → DeepStream (YOLOv8) → Local RTSP → frpc → Relay (s0/1/2)
```

**View streams:**
- http://34.47.221.242:8889/s0/ (processed s0)
- http://34.47.221.242:8889/s1/ (processed s1)
- http://34.47.221.242:8889/s2/ (processed s2)

## Scripts

| Script | Purpose |
|--------|---------|
| `system.sh` | Manage local (frpc + containers): start/stop/restart/status |
| `relay.sh` | Manage remote relay (frps + mediamtx): restart/status |
| `check.sh` | Health check (6 validation points) |
| `debug.sh` | Debug info (containers, GPU, logs) |
| `build.sh` | Rebuild Docker images |
| `start.sh` | Start containers only |
| `stop.sh` | Stop containers only |
| `frpc-*.sh` | frpc tunnel management |

## Common Tasks

### Rebuild after code changes
```bash
./build.sh            # Rebuild after editing live_stream.c
./system.sh restart   # Restart everything
```

### View logs
```bash
docker logs ds-s0 -f              # Live s0 logs
docker logs ds-s1 -f              # Live s1 logs
docker logs ds-s2 -f              # Live s2 logs
tail -f /var/log/frpc.log         # frpc tunnel logs
```

### Relay IP change
Update 3 files, then rebuild:
1. `live_stream.c` (line 97)
2. `config/frpc.ini` (lines 2, 4)
3. `publisher/loop_stream.sh` (line 7)

```bash
./build.sh
./frpc-restart.sh
./system.sh restart
```

## Troubleshooting

```bash
./check.sh            # Quick health check
./debug.sh            # Detailed debug info
./frpc-status.sh      # Check tunnel status
./relay.sh status     # Check relay status
```

**Common issues:**
- **"Resetting source" loop**: Relay not publishing to in_sX
- **frpc errors**: Wrong relay IP or token in config/frpc.ini
- **Pipeline not PLAYING**: Check docker logs for errors

## Docs

- `AGENTS.md` - Architecture and workflows
- `STANDARDS.md` - Config and coding standards
- `STRUCTURE.md` - File structure and data flow
- `relay/README.md` - Relay infrastructure setup
