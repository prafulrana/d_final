from pyservicemaker import Pipeline

import os
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer


def start_rtsp_server(rtsp_port=8554, mounts=tuple(f"/out{i}" for i in range(8)), udp_ports=tuple(5600 + i for i in range(8))):
    Gst.init(None)
    server = GstRtspServer.RTSPServer.new()
    server.props.service = str(rtsp_port)
    server.set_address("0.0.0.0")
    mounts_obj = server.get_mount_points()

    def make_factory(udp_port: int):
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_shared(True)
        launch = (
            f"( udpsrc address=127.0.0.1 port={udp_port} caps=\"application/x-rtp,"
            f"media=video,encoding-name=H264,clock-rate=90000,payload=96\" "
            f"! rtph264depay ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )"
        )
        factory.set_launch(launch)
        return factory

    for i in range(len(mounts)):
        mounts_obj.add_factory(mounts[i], make_factory(udp_ports[i]))
    server.attach(None)
    mount_list = ", ".join(mounts)
    print(f"RTSP server listening on rtsp://0.0.0.0:{rtsp_port} with mounts: {mount_list}")
    return server


def main():
    # Sample URIs cycling through available samples
    uris = [
        "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4",
        "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4",
    ]
    # Create 8 URIs by cycling through the available samples
    stream_uris = [uris[i % len(uris)] for i in range(8)]

    server = start_rtsp_server(rtsp_port=int(os.environ.get("RTSP_PORT", "8554")))

    pipeline = Pipeline("eight_stream_batch")

    pipeline.add(
        "nvmultiurisrcbin",
        "srcs",
        {
            "uri-list": ",".join(stream_uris),
            "sensor-id-list": ",".join(str(i) for i in range(8)),
            "sensor-name-list": ",".join(f"cam{i}" for i in range(8)),
            "live-source": True,
            "file-loop": True,
            "max-batch-size": 8,
            "width": 1280,
            "height": 720,
            "batched-push-timeout": 100000,
            # Pace using sender's timing; preserve upstream timestamps
            "buffer-mode": 1,  # slave
            "attach-sys-ts": True,
            # Optional: hint frame duration for NTP correction (ms)
            "frame-duration": 33,
            # Avoid default control server port 9000 conflicts
            "ip-address": "localhost",
            # Port property is a string; pass "0" to disable server fully
            "port": "0",
            # Common RTSP knobs (forwarded to rtspsrc by nvmultiurisrcbin)
            "latency": 200,
            "drop-on-latency": False,
        # Force inter-stream sync so batches are aligned across sources
        "sync-inputs": True,
        },
    )

    pgie_config = os.environ.get(
        "PGIE_CONFIG",
        "/opt/nvidia/deepstream/deepstream-8.0/pgie_b8.txt",
    )
    pipeline.add("nvinfer", "pgie", {"config-file-path": pgie_config})

    pipeline.add("nvstreamdemux", "demux")

    # Generate 8 per-stream branches: demux -> caps -> queue -> OSD -> nvvideoconvert -> NV12 caps -> encoder -> parse -> pay -> UDP
    for i in range(8):
        pipeline.add("capsfilter", f"dcaps{i}", {"caps": "video/x-raw(memory:NVMM)"})
        pipeline.add("queue", f"q{i}")
        pipeline.add("nvosdbin", f"osd{i}")
        pipeline.add("nvvideoconvert", f"conv{i}")
        pipeline.add("capsfilter", f"caps{i}", {"caps": "video/x-raw(memory:NVMM), format=NV12, framerate=30/1"})
        pipeline.add("nvv4l2h264enc", f"enc{i}", {
            "bitrate": 4000000,
            "insert-sps-pps": 1,
            "iframeinterval": int(30),
            "idrinterval": int(30),
            "intra-refresh": "0,0,0"
        })
        pipeline.add("h264parse", f"h264parse{i}")
        pipeline.add("rtph264pay", f"pay{i}", {"pt": 96})
        pipeline.add("udpsink", f"udp{i}", {"host": "127.0.0.1", "port": 5600 + i, "async": False, "sync": True})

    # Main chain (nvmultiurisrcbin already does decode + mux internally)
    pipeline.link("srcs", "pgie", "demux")

    # Link demux to all 8 stream branches
    for i in range(8):
        pipeline.link(("demux", f"dcaps{i}"), ("src_%u", ""))
        pipeline.link(f"dcaps{i}", f"q{i}", f"osd{i}", f"conv{i}", f"caps{i}", f"enc{i}", f"h264parse{i}", f"pay{i}", f"udp{i}")

    pipeline.start().wait()


if __name__ == "__main__":
    main()