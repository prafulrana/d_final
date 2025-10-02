# Plan

Goal: Minimal C, single config. Single happy path (C, master): start empty (only `/test`), add demo streams via a tiny control API that returns the RTSP URL. Python is optional for readability.

Pre‑change rule
- Always read samples in `deepstream-8.0/` before modifying code/docs. Confirmed our RTSP approach matches `deepstream_sink_bin.c` (UDP‑wrap with `rtph264pay` → `udpsink`, RTSP `udpsrc name=pay0`).

Current status
- /test works (software JPEG RTP).
- Post‑demux matches DeepStream sample pattern: per‑stream encode + RTP/UDP egress; RTSP factories wrap from UDP (no intervideo).
- OSD overlays enabled; correct order: convert → RGBA → OSD → convert → NV12 → NVENC.
- We post to nvmultiurisrcbin REST to add sources; next, we’ll wrap this behind a simple `GET /add_demo_stream` in our service.

Open items (execution plan)
1) Implement control API (single happy path)
   - Add tiny HTTP server (e.g., on `:8080`) inside `rtsp_server.c` with `GET /add_demo_stream`.
   - Handler flow:
     - Determine next index N; request `demux` pad `src_N`.
     - Build per‑stream branch and link to UDP egress at `BASE_UDP_PORT+N`.
     - Mount RTSP factory at `/sN` wrapping UDP (udpsrc name=pay0).
     - POST to nvmultiurisrcbin REST (port 9010) to add `SAMPLE_URI`.
     - Respond `200` with `{ "path": "/sN", "url": "rtsp://<PUBLIC_HOST>:<rtsp_port>/sN" }`. If N reaches 64, return HTTP 429 `{ "error": "capacity_exceeded", "max": 64 }`.
   - Startup behavior: service starts empty; only `/test` is mounted.
2) Verify end‑to‑end
   - Start empty; curl `/add_demo_stream`; receive JSON; play returned URL via ffplay.
   - Output pacing: ensure `nvmultiurisrcbin live-source=1` and UDP sink uses clock (`udpsink sync=true`) so single-stream startup does not run > realtime.
3) Keep C tiny and readable
   - Single config file; explicit pad linking; narrow API surface.
   - Minimal envs: `RTSP_PORT`, `BASE_UDP_PORT`, `SAMPLE_URI`, `PUBLIC_HOST` (no startup count; service always starts empty).
4) Engine caching — DONE
   - Persist engine to host via `/models` mount (run.sh mounts `./models` to `/models`).
   - PGIE config points `model-engine-file` to `/models/trafficcamnet_b64_gpu0_fp16.engine`.
3) Mac testability
   - Validate with ffplay over TCP from macOS; `/s0..s2` must play.

Notes
- Use C (master) for production scale. Python (`python-try`) is optional/dev.
- DeepStream REST (9000) is independent of RTSP; logs are informational.

Branch overview (what to use when)
- `master` — C, production path. NVENC→RTP/UDP→RTSP wrap. Batch‑64 by default. Engine cached under `./models`. Control API on 8080. Recommended for 64‑stream tests.
- `c-b8-config` — C, batch‑8 variant. Same as master but mux+PGIE set to 8 and `nvmultiurisrcbin port=9000` in pipeline. Use this if your GPU shows NVENC pressure with b64; you can still serve 64 streams by micro‑batching.
- `python-try` — Python GI + Flask control API (8081). Readable and close to C, but on this host hits NVENC session limits around ~8–10. Keep for dev; use C for scale.

---

 Next Phase: Scale to 64 (readiness done)

Scope: Begin with 2–3 dynamic adds via API; scale gradually.

1) RTSP parity with DeepStream samples — DONE
   - `udpsrc buffer-size` used; no `address` in RTSP factory launch.

2) Per-branch queue tuning
- DONE: `queue leaky=0`, `max-size-time=200ms`, buffers/bytes unset (0). No frame drops.

3) OSD stays ON for correctness.

4) NVENC/x264 tuning and pacing
- Keep `insert-sps-pps=1`, `idrinterval/iframeinterval=30`.
- x264: `key-int-max=30`, `tune=zerolatency`, `ultrafast`, threads default to 2 (override `X264_THREADS`).
- Pacing is handled by sink clock (`udpsink sync=true`); no videorate needed.

5) UDP ports configurability
- DONE: `BASE_UDP_PORT` env present; ensure ports starting at `BASE_UDP_PORT` are free for dynamic adds.

6) Batch and pre-demux alignment — DONE
   - In-code pre-demux: `nvmultiurisrcbin ... batched-push-timeout=33000 ! nvinfer ... ! nvstreamdemux`. Zero‑source start maintained.

7) Logging and observability
- Keep branch link logs. Optionally add lightweight per-branch FPS counters (info-level only) for soak testing.

8) Capacity checks
- Document GPU encoder session capacity and expected aggregate bandwidth (e.g., 64 × 3 Mbps ≈ 192 Mbps) in STANDARDS.md before scaling.

9) Optional codec path
- Consider H265 path parity for bandwidth reduction (doc-first; no toggles until stable).
