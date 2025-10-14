#!/usr/bin/env python3

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import argparse

# DeepStream RTSP in -> process -> RTSP out via rtspclientsink
MUXER_BATCH_TIMEOUT_USEC = 4000000  # 4 seconds for live RTSP sources

# Global state
g_pipeline = None
g_streammux = None

def bus_call(bus, message, loop):
    t = message.type

    if t == Gst.MessageType.EOS:
        print("Pipeline EOS")
        sys.stdout.flush()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        sys.stderr.flush()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        sys.stderr.flush()
    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        if struct is not None and struct.has_name("stream-eos"):
            parsed, stream_id = struct.get_uint("stream-id")
            if parsed:
                print(f"Got stream-eos from stream {stream_id} (nvurisrcbin will reconnect)")
                sys.stdout.flush()
    return True

def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad")
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name)
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

def create_source_bin(index, uri):
    print(f"Creating source bin for [{uri}]")

    # Create a source GstBin to abstract this bin's content from the rest of the pipeline
    bin_name = "source-bin-%02d" % index
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")
        return None

    # Source element - nvurisrcbin with RTSP reconnection
    uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create nvurisrcbin \n")
        return None

    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Configure RTSP reconnection
    uri_decode_bin.set_property("rtsp-reconnect-interval", 10)
    uri_decode_bin.set_property("init-rtsp-reconnect-interval", 5)
    uri_decode_bin.set_property("rtsp-reconnect-attempts", -1)  # Infinite retries
    uri_decode_bin.set_property("select-rtp-protocol", 4)  # TCP only (RTP_PROTOCOL_TCP)

    # Connect to the "pad-added" signal of nvurisrcbin which generates a
    # callback once a new pad for raw data has been created
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None

    return nbin

def main(args):
    global g_pipeline, g_streammux

    # Parse arguments
    rtsp_in = args.input
    rtsp_out = args.output
    pgie_config = args.config

    # Import probe module
    probe_module = __import__(args.probe)

    print(f"RTSP Input: {rtsp_in}")
    print(f"RTSP Output: {rtsp_out}")
    print(f"Inference config: {pgie_config}")
    print(f"Probe module: {args.probe}")

    # Initialize GStreamer
    Gst.init(None)

    # Create pipeline
    print("Creating Pipeline")
    g_pipeline = Gst.Pipeline()
    if not g_pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        return -1

    # Create nvstreammux
    print("Creating nvstreammux")
    g_streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not g_streammux:
        sys.stderr.write("Unable to create NvStreamMux\n")
        return -1

    g_pipeline.add(g_streammux)

    # Create source bin
    source_bin = create_source_bin(0, rtsp_in)
    if not source_bin:
        sys.stderr.write("Unable to create source bin\n")
        return -1

    g_pipeline.add(source_bin)

    # Link source bin to streammux
    srcpad = source_bin.get_static_pad("src")
    if not srcpad:
        sys.stderr.write("Unable to get source pad of source bin\n")
        return -1
    sinkpad = g_streammux.request_pad_simple("sink_0")
    if not sinkpad:
        sys.stderr.write("Unable to get sink pad of streammux\n")
        return -1
    if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
        sys.stderr.write("Failed to link source bin to streammux\n")
        return -1

    # Create inference
    print("Creating nvinfer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create nvinfer\n")
        return -1

    # nvsegvisual removed - using custom CUDA overlay in probe instead

    # Create OSD
    print("Creating nvdsosd")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write("Unable to create nvosd\n")
        return -1


    # Create video converter to ensure RGBA format for probe
    print("Creating nvvideoconvert")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write("Unable to create nvvideoconvert\n")
        return -1

    # Create capsfilter to force RGBA format before probe
    print("Creating RGBA capsfilter")
    rgba_caps = Gst.ElementFactory.make("capsfilter", "rgba_caps")
    if not rgba_caps:
        sys.stderr.write("Unable to create rgba_caps\n")
        return -1
    rgba_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    # Create video converter (post-OSD for format conversion to encoder)
    print("Creating nvvideoconvert-postosd")
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor-postosd")
    if not nvvidconv_postosd:
        sys.stderr.write("Unable to create nvvideoconvert-postosd\n")
        return -1

    # Create caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

    # Create encoder
    print("Creating nvv4l2h264enc")
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    if not encoder:
        sys.stderr.write("Unable to create encoder\n")
        return -1
    encoder.set_property("bitrate", 3000000)
    encoder.set_property("profile", 2)  # Main profile for better compression/smoothness
    encoder.set_property("preset-id", 0)  # P1 (highest performance preset)
    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("iframeinterval", 30)

    # Create queue after encoder
    print("Creating queue")
    queue = Gst.ElementFactory.make("queue", "queue")
    if not queue:
        sys.stderr.write("Unable to create queue\n")
        return -1

    # Create parser
    print("Creating h264parse")
    h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parse:
        sys.stderr.write("Unable to create h264parse\n")
        return -1
    h264parse.set_property("config-interval", -1)

    # Create RTSP client sink
    print("Creating rtspclientsink")
    rtsp_sink = Gst.ElementFactory.make("rtspclientsink", "rtsp-sink")
    if not rtsp_sink:
        sys.stderr.write("Unable to create rtspclientsink\n")
        return -1
    rtsp_sink.set_property("location", rtsp_out)
    rtsp_sink.set_property("protocols", 0x00000004)  # TCP only
    rtsp_sink.set_property("latency", 200)  # 200ms buffer for smooth RTP timing

    # Configure mux - set to 1280x720 to match Larix stream
    g_streammux.set_property("width", 1280)
    g_streammux.set_property("height", 720)
    g_streammux.set_property("batch-size", 1)
    g_streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    g_streammux.set_property("live-source", 1)

    # Configure inference
    pgie.set_property("config-file-path", pgie_config)

    # Add elements to pipeline
    print("Adding elements to pipeline")
    g_pipeline.add(pgie)
    g_pipeline.add(nvvidconv)
    g_pipeline.add(nvosd)
    g_pipeline.add(rgba_caps)
    g_pipeline.add(nvvidconv_postosd)
    g_pipeline.add(caps)
    g_pipeline.add(encoder)
    g_pipeline.add(queue)
    g_pipeline.add(h264parse)
    g_pipeline.add(rtsp_sink)

    # Link pipeline: pgie → nvvidconv → nvosd → rgba_caps (probe here) → nvvidconv_postosd → encoder → rtsp
    print("Linking elements")
    if not g_streammux.link(pgie):
        sys.stderr.write("Failed to link streammux → pgie\n")
        return -1
    if not pgie.link(nvvidconv):
        sys.stderr.write("Failed to link pgie → nvvidconv\n")
        return -1
    if not nvvidconv.link(nvosd):
        sys.stderr.write("Failed to link nvvidconv → nvosd\n")
        return -1
    if not nvosd.link(rgba_caps):
        sys.stderr.write("Failed to link nvosd → rgba_caps\n")
        return -1
    if not rgba_caps.link(nvvidconv_postosd):
        sys.stderr.write("Failed to link rgba_caps → nvvidconv_postosd\n")
        return -1
    if not nvvidconv_postosd.link(caps):
        sys.stderr.write("Failed to link nvvidconv_postosd → caps\n")
        return -1
    if not caps.link(encoder):
        sys.stderr.write("Failed to link caps → encoder\n")
        return -1
    if not encoder.link(queue):
        sys.stderr.write("Failed to link encoder → queue\n")
        return -1
    if not queue.link(h264parse):
        sys.stderr.write("Failed to link queue → h264parse\n")
        return -1
    if not h264parse.link(rtsp_sink):
        sys.stderr.write("Failed to link h264parse → rtsp_sink\n")
        return -1
    print("All elements linked successfully")

    # Add probe based on probe module type
    # probe_yoloworld needs to run BEFORE nvosd (to add obj_meta for nvosd to draw)
    # probe_segmentation needs to run AFTER nvosd (to access finalized segmentation metadata)
    if args.probe == "probe_yoloworld":
        # Attach probe to nvvidconv src pad (= nvosd sink pad, BEFORE nvosd processes)
        nvvidconv_srcpad = nvvidconv.get_static_pad("src")
        if nvvidconv_srcpad:
            nvvidconv_srcpad.add_probe(Gst.PadProbeType.BUFFER, probe_module.osd_sink_pad_buffer_probe, 0)
    else:
        # Attach probe to rgba_caps sink pad (AFTER nvosd)
        rgba_sinkpad = rgba_caps.get_static_pad("sink")
        if rgba_sinkpad:
            rgba_sinkpad.add_probe(Gst.PadProbeType.BUFFER, probe_module.osd_sink_pad_buffer_probe, 0)

    # Create event loop
    loop = GLib.MainLoop()
    bus = g_pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start pipeline
    print("Starting pipeline")
    g_pipeline.set_state(Gst.State.PLAYING)

    # Wait for pipeline to reach PLAYING state
    print("Waiting for state change")
    status, state, pending = g_pipeline.get_state(Gst.CLOCK_TIME_NONE)
    print(f"Current state {state}, pending state {pending}, status {status}")

    if state == Gst.State.PLAYING and pending == Gst.State.VOID_PENDING:
        print("Pipeline is PLAYING (nvurisrcbin will handle RTSP reconnection automatically)")
    else:
        print("Pipeline is not in PLAYING state")

    try:
        loop.run()
    except KeyboardInterrupt:
        pass

    # Cleanup
    print("Stopping pipeline")
    g_pipeline.set_state(Gst.State.NULL)
    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream RTSP In/Out')
    parser.add_argument("-i", "--input", required=True,
                        help="RTSP input URL (e.g., rtsp://server:8554/in_s0)")
    parser.add_argument("-o", "--output", required=True,
                        help="RTSP output URL (e.g., rtsp://server:8554/s0)")
    parser.add_argument("-c", "--config", default="/opt/nvidia/deepstream/deepstream-8.0/pgie.txt",
                        help="Path to nvinfer config file")
    parser.add_argument("--probe", default="probe_default",
                        help="Probe module name (probe_default, probe_yoloworld, etc.)")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
