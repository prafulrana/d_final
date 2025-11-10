#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
import threading
import time

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

from flask import Flask, request, jsonify

# Pipeline configuration
GPU_ID = 0
MAX_NUM_SOURCES = 36
MUXER_OUTPUT_WIDTH = 720
MUXER_OUTPUT_HEIGHT = 1280
MUXER_BATCH_TIMEOUT_USEC = 33000
PGIE_CONFIG_FILE = "/config/bowling_yolo12n_batch.txt"

# Network configuration
RTSP_SERVER_PORT = 9600
RTSP_UDPSINK_BASE_PORT = 5001
HTTP_API_PORT = 5555

# Stream restart delay
SOURCE_RESTART_DELAY_SEC = 0.5

# Global state
g_num_sources = 0
g_eos_list = [False] * MAX_NUM_SOURCES
g_source_enabled = [False] * MAX_NUM_SOURCES
g_source_bin_list = [None] * MAX_NUM_SOURCES
g_source_uris = {}
g_output_bins = [None] * MAX_NUM_SOURCES

# Pipeline elements
loop = None
pipeline = None
streammux = None
pgie = None
demux = None
rtsp_server = None

def decodebin_child_added(child_proxy, Object, name, user_data):
    """Handle decodebin child elements - set GPU ID and force TCP for RTSP"""
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    elif name.find("nvv4l2decoder") != -1:
        Object.set_property("gpu_id", GPU_ID)
    elif name.find("source") != -1:
        try:
            # Force TCP protocol for RTSP to avoid 5s UDP timeout
            Object.set_property("protocols", 0x00000004)  # GST_RTSP_LOWER_TRANS_TCP
            print(f"✓ Set {name} to TCP-only")
        except Exception:
            pass  # Not all sources support protocols property


def cb_newpad(decodebin, pad, source_id):
    """Link decoded video pad to streammux"""
    global streammux

    caps = pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()

    if gstname.find("video") != -1:
        pad_name = f"sink_{source_id}"
        sinkpad = streammux.request_pad_simple(pad_name)
        if not sinkpad:
            sys.stderr.write(f"Unable to create sink pad {pad_name}\n")
            return

        if pad.link(sinkpad) == Gst.PadLinkReturn.OK:
            print(f"Decodebin linked to pipeline: {pad_name}")
        else:
            sys.stderr.write(f"Failed to link decodebin to pipeline: {pad_name}\n")


def create_uridecode_bin(index, filename):
    """Create uridecodebin for a given source"""
    global g_source_uris, g_source_enabled

    print(f"Creating uridecodebin for [{filename}]")
    g_source_uris[index] = filename

    bin_name = f"source-bin-{index:02d}"
    bin = Gst.ElementFactory.make("uridecodebin", bin_name)
    if not bin:
        sys.stderr.write(f"Unable to create uri decode bin {bin_name}\n")
        return None

    bin.set_property("uri", filename)
    bin.connect("pad-added", cb_newpad, index)
    bin.connect("child-added", decodebin_child_added, index)
    g_source_enabled[index] = True

    return bin


def create_output_bin(source_id):
    """Create output chain: demux src_N → queue → nvvidconv → caps(RGBA) → nvosd → nvvidconv2 → caps(I420) → encoder → rtppay → udpsink"""
    global pipeline, demux, rtsp_server

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
    encoder.set_property("insert-sps-pps", True)
    encoder.set_property("idrinterval", 30)  # IDR frame every 30 frames (~1 sec at 30fps)
    encoder.set_property("iframeinterval", 0)  # Force IDR at start

    rtppay = Gst.ElementFactory.make("rtph264pay", f"rtppay-{source_id}")
    udpsink = Gst.ElementFactory.make("udpsink", f"udpsink-{source_id}")
    udpsink.set_property("host", "127.0.0.1")
    udpsink.set_property("port", RTSP_UDPSINK_BASE_PORT + source_id)
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

    # Factory is created once at startup and lives forever (like tiled version)
    # No factory management here - just return output elements
    return {"elements": elements, "demux_src": demux_src}


def stop_release_source(source_id):
    """Stop and release a source bin from the pipeline"""
    global g_num_sources, g_source_bin_list, g_output_bins, streammux, demux, pipeline

    state_return = g_source_bin_list[source_id].set_state(Gst.State.NULL)

    if state_return == Gst.StateChangeReturn.ASYNC:
        g_source_bin_list[source_id].get_state(Gst.CLOCK_TIME_NONE)

    if state_return != Gst.StateChangeReturn.FAILURE:
        # Release streammux sink pad
        pad_name = f"sink_{source_id}"
        sinkpad = streammux.get_static_pad(pad_name)
        if sinkpad:
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            streammux.release_request_pad(sinkpad)

        pipeline.remove(g_source_bin_list[source_id])
        g_num_sources -= 1
        print(f"Released source {source_id}")
    else:
        print(f"Failed to stop source {source_id}")

    # Unlink and release demux pad, keep output elements alive
    if g_output_bins[source_id]:
        try:
            demux_src = g_output_bins[source_id]["demux_src"]
            if demux_src:
                queue = g_output_bins[source_id]["elements"][0]
                queue_sink = queue.get_static_pad("sink")
                if queue_sink:
                    demux_src.unlink(queue_sink)
                demux.release_request_pad(demux_src)
                g_output_bins[source_id]["demux_src"] = None
        except Exception as e:
            print(f"Warning during pad cleanup: {e}")


def restart_source(source_id):
    """Restart a source by stopping and recreating it and its output bin"""
    global g_source_uris, g_source_enabled, g_source_bin_list, g_output_bins, pipeline, g_num_sources, rtsp_server

    if source_id < 0 or source_id >= MAX_NUM_SOURCES:
        print(f"Invalid source_id {source_id}")
        return False

    if source_id not in g_source_uris:
        print(f"Source {source_id} was never created")
        return False

    uri = g_source_uris[source_id]
    print(f"Restarting source {source_id}: {uri}")

    # Stop and release the old source
    if g_source_enabled[source_id]:
        stop_release_source(source_id)
        time.sleep(SOURCE_RESTART_DELAY_SEC)

    # Recreate the source
    source_bin = create_uridecode_bin(source_id, uri)
    if not source_bin:
        print(f"Failed to recreate source {source_id}")
        return False

    g_source_bin_list[source_id] = source_bin
    pipeline.add(source_bin)

    # Set to PLAYING
    state_return = source_bin.set_state(Gst.State.PLAYING)
    if state_return == Gst.StateChangeReturn.ASYNC:
        source_bin.get_state(Gst.CLOCK_TIME_NONE)

    # Mark source as enabled and increment count if new source
    if not g_source_enabled[source_id]:
        g_source_enabled[source_id] = True
        g_num_sources += 1
        print(f"✓ Enabled source {source_id}")

    # Relink output bin (keep elements alive, just reconnect demux pad)
    if g_output_bins[source_id] and g_output_bins[source_id].get("elements"):
        try:
            # Output bin exists, just relink demux pad
            pad_name = f"src_{source_id}"
            demux_src = demux.request_pad_simple(pad_name)
            if not demux_src:
                print(f"✗ Failed to get demux pad {pad_name}")
                return False

            queue = g_output_bins[source_id]["elements"][0]  # First element is queue
            queue_sink = queue.get_static_pad("sink")
            if not queue_sink:
                print(f"✗ Queue sink pad is None for stream {source_id}")
                return False

            if demux_src.link(queue_sink) != Gst.PadLinkReturn.OK:
                print(f"✗ Failed to relink demux to queue for stream {source_id}")
                return False

            g_output_bins[source_id]["demux_src"] = demux_src
            print(f"✓ Relinked output bin for stream {source_id}")
        except Exception as e:
            print(f"✗ Exception during pad relinking: {e}")
            return False
    else:
        # Output bin doesn't exist, create it (first time)
        output_bin = create_output_bin(source_id)
        if not output_bin:
            print(f"✗ Failed to create output for stream {source_id}")
            return False

        g_output_bins[source_id] = output_bin

        # Set output elements to PLAYING state
        for elem in output_bin["elements"]:
            state_return = elem.set_state(Gst.State.PLAYING)
            if state_return == Gst.StateChangeReturn.ASYNC:
                elem.get_state(Gst.CLOCK_TIME_NONE)

        # Create RTSP factory dynamically for this source (if not source 0)
        if source_id > 0:
            try:
                mounts = rtsp_server.get_mount_points()
                path = f"/x{source_id}"
                port = RTSP_UDPSINK_BASE_PORT + source_id
                factory = GstRtspServer.RTSPMediaFactory.new()
                launch_str = f"( udpsrc name=pay0 port={port} buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96\" )"
                factory.set_launch(launch_str)
                factory.set_shared(True)
                mounts.add_factory(path, factory)
                print(f"✓ RTSP factory created at {path} (listening on UDP {port})")
            except Exception as e:
                print(f"⚠ Failed to create RTSP factory for {path}: {e}")

    print(f"✓ Restarted source {source_id}")
    return True

def bus_call(bus, message, loop):
    global g_eos_list
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream (all sources disconnected, keeping service alive for manual restart)\n")
        # Don't quit - keep HTTP API available for stream restart via /stream/restart endpoint
        # loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        # Don't quit on errors - keep service alive for restart API
        # All errors are considered non-fatal to keep HTTP API available
        src = message.src
        if src:
            sys.stderr.write(f"Error from {src.get_name()} (non-fatal), continuing...\n")
        else:
            sys.stderr.write(f"Pipeline error (non-fatal), continuing...\n")
    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        #Check for stream-eos message
        if struct is not None and struct.has_name("stream-eos"):
            parsed, stream_id = struct.get_uint("stream-id")
            if parsed:
                #Set eos status of stream to True, to be deleted in delete-sources
                print("Got EOS from stream %d" % stream_id)
                g_eos_list[stream_id] = True
    return True

# Flask HTTP control server
app = Flask(__name__)

@app.route('/stream/restart', methods=['POST'])
def http_restart_stream():
    data = request.json
    source_id = data.get('id')

    if source_id is None:
        return jsonify({"status": "error", "message": "id required"}), 400

    if restart_source(source_id):
        return jsonify({"status": "ok", "id": source_id})
    else:
        return jsonify({"status": "error", "message": "restart failed"}), 500

@app.route('/stream/status', methods=['GET'])
def http_status():
    active = [i for i in range(MAX_NUM_SOURCES) if g_source_enabled[i]]
    uris = {i: g_source_uris.get(i, "") for i in active}
    rtsp_paths = {i: f"/x{i}" for i in active}
    return jsonify({
        "active_streams": active,
        "count": len(active),
        "uris": uris,
        "rtsp_server": f"rtsp://localhost:{RTSP_SERVER_PORT}",
        "rtsp_paths": rtsp_paths
    })

def run_flask():
    app.run(host='0.0.0.0', port=HTTP_API_PORT, threaded=True)

def main(args):
    global g_num_sources, g_source_bin_list, g_output_bins
    global loop, pipeline, streammux, pgie, demux, rtsp_server

    if len(args) < 2:
        sys.stderr.write(f"usage: {args[0]} <uri1> [uri2] [uri3] ...\n")
        sys.exit(1)

    total_sources = len(args) - 1

    Gst.init(None)

    print("Creating pipeline")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Unable to create pipeline\n")
        sys.exit(1)

    print("Creating streammux")
    is_live = False

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create streammux\n")
        sys.exit(1)

    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("batch-size", MAX_NUM_SOURCES)
    streammux.set_property("gpu_id", GPU_ID)
    streammux.set_property("live-source", 1)
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)

    pipeline.add(streammux)

    # Store all URIs for restart API
    for i in range(total_sources):
        uri_name = args[i+1]
        g_source_uris[i] = uri_name
        if uri_name.find("rtsp://") == 0:
            is_live = True

    # Create first source at startup
    print("Creating source_bin 0")
    source_bin = create_uridecode_bin(0, args[1])
    if not source_bin:
        sys.stderr.write("Failed to create source bin 0\n")
        sys.exit(1)
    g_source_bin_list[0] = source_bin
    pipeline.add(source_bin)

    num_sources = 1
    g_num_sources = 1

    print("Creating pgie")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create pgie\n")
        sys.exit(1)
    pgie.set_property("config-file-path", PGIE_CONFIG_FILE)
    pgie.set_property("batch-size", MAX_NUM_SOURCES)
    pgie.set_property("gpu_id", GPU_ID)

    print("Creating demux")
    demux = Gst.ElementFactory.make("nvstreamdemux", "demuxer")
    if not demux:
        sys.stderr.write("Unable to create demux\n")
        sys.exit(1)

    pipeline.add(pgie)
    pipeline.add(demux)

    print("Linking pipeline: streammux -> pgie -> demux")
    streammux.link(pgie)
    pgie.link(demux)

    # Create RTSP server
    print(f"Creating RTSP server on port {RTSP_SERVER_PORT}")
    rtsp_server = GstRtspServer.RTSPServer.new()
    rtsp_server.props.service = str(RTSP_SERVER_PORT)
    rtsp_server.attach(None)

    # Create RTSP factory for first source (others created on-demand)
    mounts = rtsp_server.get_mount_points()
    factory = GstRtspServer.RTSPMediaFactory.new()
    launch_str = f"( udpsrc name=pay0 port={RTSP_UDPSINK_BASE_PORT} buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96\" )"
    factory.set_launch(launch_str)
    factory.set_shared(True)
    mounts.add_factory("/x0", factory)
    print(f"✓ RTSP factory created at /x0 (UDP port {RTSP_UDPSINK_BASE_PORT})")

    # Create output bin for first source
    print("Creating output bin for source 0")
    output_bin = create_output_bin(0)
    if not output_bin:
        sys.stderr.write("Failed to create output bin for source 0\n")
        sys.exit(1)
    g_output_bins[0] = output_bin

    # Start Flask HTTP API
    print(f"Starting HTTP control server on port {HTTP_API_PORT}")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Setup GLib event loop and bus
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start pipeline
    pipeline.set_state(Gst.State.PAUSED)
    print(f"Now playing: {args[1]} -> rtsp://localhost:{RTSP_SERVER_PORT}/x0")
    pipeline.set_state(Gst.State.PLAYING)

    # Wait for pipeline to reach PLAYING state
    status, state, pending = pipeline.get_state(Gst.CLOCK_TIME_NONE)

    # Sync output elements with pipeline
    if g_output_bins[0]:
        for elem in g_output_bins[0]["elements"]:
            elem.sync_state_with_parent()

    print("\n" + "="*70)
    print("✓ Pipeline ready")
    print(f"  Active sources: {num_sources}/{MAX_NUM_SOURCES}")
    print(f"  Batch size: {MAX_NUM_SOURCES}")
    print(f"\nRTSP Server: rtsp://localhost:{RTSP_SERVER_PORT}")
    print(f"  Stream 0 (in_s0): /x0")
    print(f"\nHTTP Control API: http://localhost:{HTTP_API_PORT}")
    print(f"  POST /stream/restart {{\"id\": N}}")
    print(f"  GET  /stream/status")
    print("="*70 + "\n")

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("Shutting down pipeline")
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
