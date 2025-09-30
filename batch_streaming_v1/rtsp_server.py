#!/usr/bin/env python3
"""
RTSP server that wraps UDP streams (from DeepStream pipeline) as RTSP endpoints.
Mounts N streams as rtsp://host:8554/s0 through /s(N-1)
"""
import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

PORT_BASE = 5600

class RtspPostDemux:
    def __init__(self, pipeline_desc: str, streams: int = 30):
        Gst.init(None)
        self.loop = GLib.MainLoop()
        self.pipeline = Gst.parse_launch(pipeline_desc)
        if not isinstance(self.pipeline, Gst.Pipeline):
            top = Gst.Pipeline.new(None)
            top.add(self.pipeline)
            self.pipeline = top

        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")
        self.mounts = self.server.get_mount_points()
        self.server.attach(None)

        for idx in range(streams):
            self._mount_rtsp_for_port(PORT_BASE + idx, idx)

    def _mount_rtsp_for_port(self, port: int, idx: int):
        path = f"/s{idx}"
        caps = (
            'application/x-rtp,media=video,encoding-name=H264,'
            'payload=96,clock-rate=90000'
        )
        launch = (
            f"( udpsrc address=127.0.0.1 port={port} caps=\"{caps}\" "
            f"! rtph264depay ! h264parse ! rtph264pay config-interval=1 pt=96 name=pay0 )"
        )
        f = GstRtspServer.RTSPMediaFactory()
        f.set_launch(launch)
        f.set_shared(True)
        self.mounts.add_factory(path, f)
        print(f"RTSP mounted: rtsp://127.0.0.1:8554{path} (from UDP {port})")

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        print('Pipeline PLAYING. RTSP server on 8554.')
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        self.pipeline.set_state(Gst.State.NULL)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} config.txt", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        pipeline_desc = f.read().strip()
    app = RtspPostDemux(pipeline_desc, streams=4)
    app.run()


if __name__ == '__main__':
    main()