#!/usr/bin/env python3

import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True

def main():
    Gst.init(None)
    sys.stdout.write("Starting DeepStream RTSP pipeline for s0...\n")
    sys.stdout.flush()

    # Create Pipeline
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        return

    # Source: nvurisrcbin
    source = Gst.ElementFactory.make("nvurisrcbin", "uri-source")
    source.set_property('uri', 'rtsp://34.14.140.30:8554/in_s0')
    source.set_property('rtsp-reconnect-interval', 10)
    source.set_property('latency', 2000)
    source.set_property('select-rtp-protocol', 4)  # TCP

    # Streammux
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 33333)
    streammux.set_property('live-source', 0)

    # Primary GIE (inference)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "/config/config_infer_primary.txt")

    # nvvidconv before OSD
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")

    # OSD (matches config/config_osd.txt settings)
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvosd.set_property('display-text', 1)
    nvosd.set_property('display-bbox', 1)
    nvosd.set_property('process-mode', 1)
    nvosd.set_property('gpu-id', 0)

    # nvvidconv after OSD
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")

    # Caps filter - NV12 format
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    # Encoder
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    encoder.set_property('bitrate', 2000000)
    encoder.set_property('iframeinterval', 30)

    # Queue
    queue = Gst.ElementFactory.make("queue", "queue")

    # H264 parser
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    h264parser.set_property('config-interval', -1)

    # RTP payloader
    rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")

    # UDP sink
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    sink.set_property('host', '127.0.0.1')
    sink.set_property('port', 5400)
    sink.set_property('sync', 1)

    # Add elements to pipeline
    pipeline.add(source)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(queue)
    pipeline.add(h264parser)
    pipeline.add(rtppay)
    pipeline.add(sink)

    # Dynamic pad handling for nvurisrcbin
    def on_pad_added(src, new_pad):
        sys.stdout.write("Received new pad '%s' from '%s'\n" % (new_pad.get_name(), src.get_name()))
        sys.stdout.flush()
        sink_pad = streammux.get_request_pad("sink_0")
        if sink_pad and not sink_pad.is_linked():
            ret = new_pad.link(sink_pad)
            sys.stdout.write("Link result: %s\n" % ret)
            sys.stdout.flush()

    source.connect("pad-added", on_pad_added)

    # Link rest of pipeline
    if not streammux.link(pgie):
        sys.stderr.write("Failed to link streammux to pgie\n")
        return
    if not pgie.link(nvvidconv):
        sys.stderr.write("Failed to link pgie to nvvidconv\n")
        return
    if not nvvidconv.link(nvosd):
        sys.stderr.write("Failed to link nvvidconv to nvosd\n")
        return
    if not nvosd.link(nvvidconv_postosd):
        sys.stderr.write("Failed to link nvosd to nvvidconv_postosd\n")
        return
    if not nvvidconv_postosd.link(caps):
        sys.stderr.write("Failed to link nvvidconv_postosd to caps\n")
        return
    if not caps.link(encoder):
        sys.stderr.write("Failed to link caps to encoder\n")
        return
    if not encoder.link(queue):
        sys.stderr.write("Failed to link encoder to queue\n")
        return
    if not queue.link(h264parser):
        sys.stderr.write("Failed to link queue to h264parser\n")
        return
    if not h264parser.link(rtppay):
        sys.stderr.write("Failed to link h264parser to rtppay\n")
        return
    if not rtppay.link(sink):
        sys.stderr.write("Failed to link rtppay to sink\n")
        return

    sys.stdout.write("\n *** Pipeline linked successfully ***\n\n")
    sys.stdout.flush()

    # Create RTSP server
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "8554"
    server.attach(None)
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch("( udpsrc name=pay0 port=5400 buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)H264, payload=96\" )")
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)
    sys.stdout.write("\n *** RTSP Server ready at rtsp://localhost:8554/ds-test ***\n\n")
    sys.stdout.flush()

    # Event loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start play
    pipeline.set_state(Gst.State.PLAYING)
    sys.stdout.write("\n *** Pipeline set to PLAYING ***\n\n")
    sys.stdout.flush()

    try:
        loop.run()
    except:
        pass

    # Cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main())
