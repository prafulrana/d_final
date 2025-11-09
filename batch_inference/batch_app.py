#!/usr/bin/env python3

import sys
import gi
import threading
import configparser
from flask import Flask, request, jsonify

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

# Configuration
MAX_NUM_SOURCES = 36
GPU_ID = 0
MUXER_OUTPUT_WIDTH = 720
MUXER_OUTPUT_HEIGHT = 1280
MUXER_BATCH_TIMEOUT_USEC = 33000
PGIE_CONFIG_FILE = "/config/bowling_yolo12n_batch.txt"
TRACKER_CONFIG_FILE = "/config/tracker_batch.yml"

# Global state
g_source_bins = [None] * MAX_NUM_SOURCES
g_source_enabled = [False] * MAX_NUM_SOURCES
g_output_bins = [None] * MAX_NUM_SOURCES
g_rtsp_servers = [None] * MAX_NUM_SOURCES
g_num_sources = 0

loop = None
pipeline = None
streammux = None
demux = None

def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if name.find("nvv4l2decoder") != -1:
        Object.set_property("gpu_id", GPU_ID)

def cb_newpad(decodebin, pad, data):
    global streammux
    caps = pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()

    if gstname.find("video") != -1:
        source_id = data
        pad_name = "sink_%u" % source_id
        sinkpad = streammux.request_pad_simple(pad_name)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin\n")
            return
        if pad.link(sinkpad) == Gst.PadLinkReturn.OK:
            print(f"✓ Stream {source_id} linked to mux")
        else:
            sys.stderr.write("Failed to link decodebin to pipeline\n")

def create_source_bin(source_id, uri):
    bin_name = f"source-bin-{source_id:02d}"
    bin = Gst.ElementFactory.make("uridecodebin", bin_name)
    if not bin:
        sys.stderr.write("Unable to create uridecodebin\n")
        return None

    bin.set_property("uri", uri)
    bin.connect("pad-added", cb_newpad, source_id)
    bin.connect("child-added", decodebin_child_added, source_id)

    return bin

def create_output_bin(source_id):
    """Create output chain: demux src_N → queue → nvvidconv → nvosd → encoder → rtppay → udpsink"""
    global pipeline, demux, g_rtsp_servers

    # Request demux src pad
    pad_name = f"src_{source_id}"
    demux_src = demux.request_pad_simple(pad_name)
    if not demux_src:
        print(f"✗ Failed to get demux pad {pad_name}")
        return None

    # Create output elements
    queue = Gst.ElementFactory.make("queue", f"queue-{source_id}")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv-{source_id}")
    caps_filter = Gst.ElementFactory.make("capsfilter", f"caps-{source_id}")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    caps_filter.set_property("caps", caps)

    nvosd = Gst.ElementFactory.make("nvdsosd", f"nvosd-{source_id}")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv2-{source_id}")
    caps_filter2 = Gst.ElementFactory.make("capsfilter", f"caps2-{source_id}")
    caps2 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    caps_filter2.set_property("caps", caps2)

    encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"encoder-{source_id}")
    encoder.set_property("bitrate", 4000000)

    rtppay = Gst.ElementFactory.make("rtph264pay", f"rtppay-{source_id}")
    udpsink = Gst.ElementFactory.make("udpsink", f"udpsink-{source_id}")
    udpsink.set_property("host", "127.0.0.1")
    udpsink.set_property("port", 5000 + source_id)  # Temp port for RTSP server
    udpsink.set_property("async", False)
    udpsink.set_property("sync", 0)

    elements = [queue, nvvidconv, caps_filter, nvosd, nvvidconv2, caps_filter2, encoder, rtppay, udpsink]

    # Add to pipeline
    for elem in elements:
        pipeline.add(elem)

    # Link elements
    queue_sink = queue.get_static_pad("sink")
    if demux_src.link(queue_sink) != Gst.PadLinkReturn.OK:
        print(f"✗ Failed to link demux to queue for stream {source_id}")
        return None

    # Link output chain element by element
    if not queue.link(nvvidconv):
        print(f"✗ Failed to link queue to nvvidconv for stream {source_id}")
        return None
    if not nvvidconv.link(caps_filter):
        print(f"✗ Failed to link nvvidconv to caps for stream {source_id}")
        return None
    if not caps_filter.link(nvosd):
        print(f"✗ Failed to link caps to nvosd for stream {source_id}")
        return None
    if not nvosd.link(nvvidconv2):
        print(f"✗ Failed to link nvosd to nvvidconv2 for stream {source_id}")
        return None
    if not nvvidconv2.link(caps_filter2):
        print(f"✗ Failed to link nvvidconv2 to caps2 for stream {source_id}")
        return None
    if not caps_filter2.link(encoder):
        print(f"✗ Failed to link caps2 to encoder for stream {source_id}")
        return None
    if not encoder.link(rtppay):
        print(f"✗ Failed to link encoder to rtppay for stream {source_id}")
        return None
    if not rtppay.link(udpsink):
        print(f"✗ Failed to link rtppay to udpsink for stream {source_id}")
        return None

    # Reuse existing RTSP server if it exists, otherwise create new one
    if g_rtsp_servers[source_id]:
        server = g_rtsp_servers[source_id]
        print(f"✓ Reusing existing RTSP server at rtsp://localhost:{8554 + source_id}/ds-test")
    else:
        rtsp_port = 8554 + source_id
        server = GstRtspServer.RTSPServer.new()
        server.props.service = str(rtsp_port)

        factory = GstRtspServer.RTSPMediaFactory.new()
        launch_str = f"( udpsrc name=pay0 port={5000 + source_id} buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96\" )"
        factory.set_launch(launch_str)
        factory.set_shared(True)

        mounts = server.get_mount_points()
        mounts.add_factory("/ds-test", factory)
        server.attach(None)

        print(f"✓ RTSP server ready at rtsp://localhost:{rtsp_port}/ds-test")

    return {"elements": elements, "demux_src": demux_src, "server": server}

def add_source(source_id, uri):
    global g_num_sources, g_source_enabled, g_source_bins, g_output_bins, g_rtsp_servers, pipeline

    if source_id < 0 or source_id >= MAX_NUM_SOURCES:
        print(f"✗ Invalid source_id {source_id}")
        return False

    if g_source_enabled[source_id]:
        print(f"Stream {source_id} already active")
        return False

    print(f"Adding stream {source_id}: {uri}")

    # Create source bin
    source_bin = create_source_bin(source_id, uri)
    if not source_bin:
        return False

    g_source_bins[source_id] = source_bin
    pipeline.add(source_bin)

    # Set to PLAYING
    state_return = source_bin.set_state(Gst.State.PLAYING)
    if state_return == Gst.StateChangeReturn.ASYNC:
        source_bin.get_state(Gst.CLOCK_TIME_NONE)

    # Create output bin (demux → RTSP)
    output_bin = create_output_bin(source_id)
    if not output_bin:
        print(f"✗ Failed to create output for stream {source_id}")
        return False

    g_output_bins[source_id] = output_bin
    g_rtsp_servers[source_id] = output_bin["server"]

    # Sync output elements state with pipeline
    for elem in output_bin["elements"]:
        elem.sync_state_with_parent()

    g_source_enabled[source_id] = True
    g_num_sources += 1

    print(f"✓ Added stream {source_id}, total sources: {g_num_sources}")
    return True

def remove_source(source_id):
    global g_num_sources, g_source_enabled, g_source_bins, g_output_bins, pipeline, streammux, demux

    if source_id < 0 or source_id >= MAX_NUM_SOURCES:
        return False

    if not g_source_enabled[source_id]:
        print(f"Stream {source_id} not active")
        return False

    print(f"Removing stream {source_id}")

    # Stop and remove source bin
    if g_source_bins[source_id]:
        g_source_bins[source_id].set_state(Gst.State.NULL)
        pad_name = f"sink_{source_id}"
        sinkpad = streammux.get_static_pad(pad_name)
        if sinkpad:
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            streammux.release_request_pad(sinkpad)
        pipeline.remove(g_source_bins[source_id])
        g_source_bins[source_id] = None

    # Stop and remove output bin
    if g_output_bins[source_id]:
        for elem in g_output_bins[source_id]["elements"]:
            elem.set_state(Gst.State.NULL)
            pipeline.remove(elem)
        demux_src = g_output_bins[source_id]["demux_src"]
        demux.release_request_pad(demux_src)
        g_output_bins[source_id] = None

    # Note: We don't cleanup the RTSP server - it can stay running
    # The udpsrc in the RTSP factory will just not receive data until
    # the stream is added back and the output pipeline is recreated

    g_source_enabled[source_id] = False
    g_num_sources -= 1

    print(f"✓ Removed stream {source_id}, total sources: {g_num_sources}")
    return True

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write(f"Warning: {err}: {debug}\n")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"Error: {err}: {debug}\n")
        # Don't quit on source-level errors, only pipeline-level
        src = message.src
        if src and ("source-bin" in src.get_name() or "uridecodebin" in src.get_name()):
            sys.stderr.write(f"Stream source error (non-fatal), continuing...\n")
        else:
            loop.quit()
    return True

# Flask HTTP control server
app = Flask(__name__)

@app.route('/stream/add', methods=['POST'])
def http_add_stream():
    data = request.json
    source_id = data.get('id')
    uri = data.get('uri', f"rtsp://34.47.221.242:8554/in_s{source_id}")

    if add_source(source_id, uri):
        return jsonify({"status": "ok", "id": source_id})
    else:
        return jsonify({"status": "error"}), 400

@app.route('/stream/remove', methods=['POST'])
def http_remove_stream():
    data = request.json
    source_id = data.get('id')

    if remove_source(source_id):
        return jsonify({"status": "ok", "id": source_id})
    else:
        return jsonify({"status": "error"}), 400

@app.route('/stream/status', methods=['GET'])
def http_status():
    active = [i for i in range(MAX_NUM_SOURCES) if g_source_enabled[i]]
    return jsonify({"active_streams": active, "count": g_num_sources})

def run_flask():
    app.run(host='0.0.0.0', port=5555, threaded=True)

def main():
    global loop, pipeline, streammux, demux

    Gst.init(None)

    print("Creating pipeline...")
    pipeline = Gst.Pipeline()

    # Create nvstreammux
    print("Creating nvstreammux...")
    streammux = Gst.ElementFactory.make("nvstreammux", "muxer")
    if not streammux:
        sys.stderr.write("Unable to create nvstreammux\n")
        sys.exit(1)

    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("batch-size", MAX_NUM_SOURCES)
    streammux.set_property("gpu_id", GPU_ID)
    streammux.set_property("live-source", 1)
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)

    # Create PGIE
    print("Creating PGIE...")
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    if not pgie:
        sys.stderr.write("Unable to create pgie\n")
        sys.exit(1)

    pgie.set_property("config-file-path", PGIE_CONFIG_FILE)
    pgie.set_property("batch-size", MAX_NUM_SOURCES)
    pgie.set_property("gpu_id", GPU_ID)

    # Create tracker
    print("Creating tracker...")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write("Unable to create tracker\n")
        sys.exit(1)

    tracker.set_property("tracker-width", MUXER_OUTPUT_WIDTH)
    tracker.set_property("tracker-height", MUXER_OUTPUT_HEIGHT)
    tracker.set_property("gpu_id", GPU_ID)
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", TRACKER_CONFIG_FILE)

    # Create nvstreamdemux
    print("Creating nvstreamdemux...")
    demux = Gst.ElementFactory.make("nvstreamdemux", "demuxer")
    if not demux:
        sys.stderr.write("Unable to create nvstreamdemux\n")
        sys.exit(1)

    # Add to pipeline and link
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(demux)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(demux)

    # Create event loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start Flask in separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start pipeline
    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    print("\n" + "="*60)
    print("✓ Batch inference pipeline ready")
    print(f"  Max sources: {MAX_NUM_SOURCES}")
    print(f"  Batch size: {MAX_NUM_SOURCES}")
    print(f"  Batch timeout: {MUXER_BATCH_TIMEOUT_USEC}us")
    print("\nHTTP Control API (port 5555):")
    print("  POST /stream/add    {\"id\": 0}")
    print("  POST /stream/remove {\"id\": 0}")
    print("  GET  /stream/status")
    print("="*60 + "\n")

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nExiting...")

    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main())
