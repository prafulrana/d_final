# Plan

Goal: Minimal C, single config. Post‑demux work stays simple; C only boots the pipeline and exposes RTSP mounts that are easy to test from macOS.

Pre‑change rule
- Always read samples in `deepstream-8.0/` before modifying code/docs. Confirmed our RTSP approach matches `deepstream_sink_bin.c` (UDP‑wrap with `rtph264pay` → `udpsink`, RTSP `udpsrc name=pay0`).

Current status
- /test works (software JPEG RTP).
- Post‑demux matches DeepStream sample pattern: per‑stream encode + RTP/UDP egress; RTSP factories wrap from UDP (no intervideo).
- OSD overlays enabled; correct order: convert → RGBA → OSD → convert → NV12 → NVENC.

Open items (execution plan)
1) Verify /s0..s1 playback via UDP‑wrapped RTSP
   - Branch (per stream): queue (leaky, 200ms) → nvvidconv → caps NVMM RGBA → nvosdbin → nvvidconv → caps NVMM NV12 → nvv4l2h264enc → h264parse → rtph264pay → udpsink:127.0.0.1:(BASE_UDP_PORT+i)
   - RTSP factory: `( udpsrc port=BASE_UDP_PORT+i buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" name=pay0 )`
   - Mirrors `deepstream_sink_bin.c` UDP-wrap approach.
2) Keep C tiny and readable
   - Parse only `pipeline.txt`, build/link branches, start RTSP.
   - Minimal envs: `STREAMS`, `RTSP_PORT`, `BASE_UDP_PORT`, `USE_OSD`.
3) Mac testability
   - Validate with ffplay over TCP from macOS; `/s0..s2` must play.

Notes
- We intentionally avoid Python. All serving is C + config.
- If DeepStream REST (port 9000) is occupied in your environment, it does not affect RTSP; warnings are benign.

---

Next Phase: Scale to 64 (readiness checklist)

Scope: Keep STREAMS=2 for now. Do not implement yet; capture changes to apply before scaling tests.

1) RTSP parity with DeepStream samples
- DONE: `udpsrc buffer-size` used; no `address` in RTSP factory launch.

2) Per-branch queue tuning
- DONE: `queue leaky=2`, `max-size-time=200ms`, buffers/bytes unset (0).

3) Default OSD off for scale
- Consider flipping `USE_OSD=0` only when scaling soak tests; current default remains on for correctness.

4) NVENC tuning and bitrate
- Keep `insert-sps-pps=1`, `idrinterval/iframeinterval` aligned to framerate. Consider lower per-stream bitrate (e.g., 2–3 Mbps @720p30) with a single env for consistency.

5) UDP ports configurability
- DONE: `BASE_UDP_PORT` env present; ensure `[BASE_UDP_PORT .. +STREAMS-1]` is free.

6) Batch and pre-demux alignment
- Require `pipeline.txt max-batch-size == STREAMS`; confirm framerate/resize are set pre-demux to keep NVENC input uniform.

7) Logging and observability
- Keep branch link logs. Optionally add lightweight per-branch FPS counters (info-level only) for soak testing.

8) Capacity checks
- Document GPU encoder session capacity and expected aggregate bandwidth (e.g., 64 × 3 Mbps ≈ 192 Mbps) in STANDARDS.md before scaling.

9) Optional codec path
- Consider H265 path parity for bandwidth reduction (doc-first; no toggles until stable).
