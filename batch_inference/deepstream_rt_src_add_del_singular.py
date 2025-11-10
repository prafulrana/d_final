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
import gi
import configparser
import threading
from flask import Flask, request, jsonify
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer
from ctypes import *
import time
import math
import random
import platform

import pyds

MAX_DISPLAY_LEN=64
PGIE_CLASS_ID_BALL = 0
PGIE_CLASS_ID_PINS = 1
PGIE_CLASS_ID_SWEEP = 2
MUXER_OUTPUT_WIDTH=720
MUXER_OUTPUT_HEIGHT=1280
MUXER_BATCH_TIMEOUT_USEC = 33000
GPU_ID = 0
MAX_NUM_SOURCES = 36
PGIE_CONFIG_FILE = "/config/bowling_yolo12n_batch.txt"

CONFIG_GPU_ID = "gpu-id"

g_num_sources = 0
g_source_id_list = [0] * MAX_NUM_SOURCES
g_eos_list = [False] * MAX_NUM_SOURCES
g_source_enabled = [False] * MAX_NUM_SOURCES
g_source_bin_list = [None] * MAX_NUM_SOURCES
g_source_uris = {}  # Store original URIs for each source
g_output_bins = [None] * MAX_NUM_SOURCES
g_rtsp_servers = [None] * MAX_NUM_SOURCES

pgie_classes_str= ["bowling-ball", "bowling-pins", "sweep-board"]

uri = ""

loop = None
pipeline = None
streammux = None
pgie = None
demux = None
rtsp_server = None  # Single RTSP server for all streams

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)
    if(name.find("nvv4l2decoder") != -1):
        Object.set_property("gpu_id", GPU_ID)


def cb_newpad(decodebin,pad,data):
    global streammux
    print("In cb_newpad\n")
    caps=pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        source_id = data
        pad_name = "sink_%u" % source_id
        print(pad_name)
        #Get a sink pad from the streammux, link to decodebin
        sinkpad = streammux.request_pad_simple(pad_name)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        if pad.link(sinkpad) == Gst.PadLinkReturn.OK:
            print("Decodebin linked to pipeline")
        else:
            sys.stderr.write("Failed to link decodebin to pipeline\n")


def create_uridecode_bin(index,filename):
    global g_source_id_list
    global g_source_uris
    print("Creating uridecodebin for [%s]" % filename)

    # Store the URI for this source
    g_source_uris[index] = filename

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    g_source_id_list[index] = index
    bin_name="source-bin-%02d" % index
    print(bin_name)

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    bin=Gst.ElementFactory.make("uridecodebin", bin_name)
    if not bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    bin.set_property("uri",filename)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has been created by the decodebin
    bin.connect("pad-added",cb_newpad,g_source_id_list[index])
    bin.connect("child-added",decodebin_child_added,g_source_id_list[index])

    #Set status of the source to enabled
    g_source_enabled[index] = True

    return bin


def create_output_bin(source_id):
    """Create output chain: demux src_N → queue → nvvidconv → caps(RGBA) → nvosd → nvvidconv2 → caps(I420) → encoder → rtppay → udpsink"""
    global pipeline, demux, g_rtsp_servers, rtsp_server

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

    rtppay = Gst.ElementFactory.make("rtph264pay", f"rtppay-{source_id}")
    udpsink = Gst.ElementFactory.make("udpsink", f"udpsink-{source_id}")
    udpsink.set_property("host", "127.0.0.1")
    udpsink.set_property("port", 5001 + source_id)  # Port for RTSP udpsrc
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

    # Add factory to global RTSP server with path /x{source_id+1}
    path = f"/x{source_id + 1}"
    factory = GstRtspServer.RTSPMediaFactory.new()
    launch_str = f"( udpsrc name=pay0 port={5001 + source_id} buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96\" )"
    factory.set_launch(launch_str)
    factory.set_shared(True)

    mounts = rtsp_server.get_mount_points()
    mounts.add_factory(path, factory)

    print(f"✓ RTSP path ready at rtsp://localhost:9600{path}")

    return {"elements": elements, "demux_src": demux_src, "factory": factory}


def stop_release_source(source_id):
    global g_num_sources
    global g_source_bin_list
    global g_output_bins
    global streammux
    global demux
    global pipeline

    #Attempt to change status of source to be released
    state_return = g_source_bin_list[source_id].set_state(Gst.State.NULL)

    if state_return == Gst.StateChangeReturn.SUCCESS:
        print("STATE CHANGE SUCCESS\n")
        pad_name = "sink_%u" % source_id
        print(pad_name)
        #Retrieve sink pad to be released
        sinkpad = streammux.get_static_pad(pad_name)
        #Send flush stop event to the sink pad, then release from the streammux
        sinkpad.send_event(Gst.Event.new_flush_stop(False))
        streammux.release_request_pad(sinkpad)
        print("STATE CHANGE SUCCESS\n")
        #Remove the source bin from the pipeline
        pipeline.remove(g_source_bin_list[source_id])
        g_num_sources -= 1

    elif state_return == Gst.StateChangeReturn.FAILURE:
        print("STATE CHANGE FAILURE\n")

    elif state_return == Gst.StateChangeReturn.ASYNC:
        state_return = g_source_bin_list[source_id].get_state(Gst.CLOCK_TIME_NONE)
        pad_name = "sink_%u" % source_id
        print(pad_name)
        sinkpad = streammux.get_static_pad(pad_name)
        sinkpad.send_event(Gst.Event.new_flush_stop(False))
        streammux.release_request_pad(sinkpad)
        print("STATE CHANGE ASYNC\n")
        pipeline.remove(g_source_bin_list[source_id])
        g_num_sources -= 1

    # Stop and remove output bin
    if g_output_bins[source_id]:
        for elem in g_output_bins[source_id]["elements"]:
            elem.set_state(Gst.State.NULL)
            pipeline.remove(elem)
        demux_src = g_output_bins[source_id]["demux_src"]
        demux.release_request_pad(demux_src)
        g_output_bins[source_id] = None


def delete_sources(data):
    global loop
    global g_num_sources
    global g_eos_list
    global g_source_enabled

    #First delete sources that have reached end of stream
    for source_id in range(MAX_NUM_SOURCES):
        if (g_eos_list[source_id] and g_source_enabled[source_id]):
            g_source_enabled[source_id] = False
            stop_release_source(source_id)

    #Quit if no sources remaining
    if (g_num_sources == 0):
        loop.quit()
        print("All sources stopped quitting")
        return False

    #Randomly choose an enabled source to delete
    source_id = random.randrange(0, MAX_NUM_SOURCES)
    while (not g_source_enabled[source_id]):
        source_id = random.randrange(0, MAX_NUM_SOURCES)
    #Disable the source
    g_source_enabled[source_id] = False
    #Release the source
    print("Calling Stop %d " % source_id)
    stop_release_source(source_id)

    #Quit if no sources remaining
    if (g_num_sources == 0):
        loop.quit()
        print("All sources stopped quitting")
        return False

    return True


def add_sources(data):
    global g_num_sources
    global g_source_enabled
    global g_source_bin_list
    global pipeline

    source_id = g_num_sources

    #Randomly select an un-enabled source to add
    source_id = random.randrange(0, MAX_NUM_SOURCES)
    while (g_source_enabled[source_id]):
        source_id = random.randrange(0, MAX_NUM_SOURCES)

    #Enable the source
    g_source_enabled[source_id] = True

    print("Calling Start %d " % source_id)

    #Create a uridecode bin with the chosen source id
    source_bin = create_uridecode_bin(source_id, uri)

    if (not source_bin):
        sys.stderr.write("Failed to create source bin. Exiting.")
        exit(1)

    #Add source bin to our list and to pipeline
    g_source_bin_list[source_id] = source_bin
    pipeline.add(source_bin)

    #Set state of source bin to playing
    state_return = g_source_bin_list[source_id].set_state(Gst.State.PLAYING)

    if state_return == Gst.StateChangeReturn.SUCCESS:
        print("STATE CHANGE SUCCESS\n")
        source_id += 1

    elif state_return == Gst.StateChangeReturn.FAILURE:
        print("STATE CHANGE FAILURE\n")

    elif state_return == Gst.StateChangeReturn.ASYNC:
        state_return = g_source_bin_list[source_id].get_state(Gst.CLOCK_TIME_NONE)
        source_id += 1

    elif state_return == Gst.StateChangeReturn.NO_PREROLL:
        print("STATE CHANGE NO PREROLL\n")

    g_num_sources += 1

    #If reached the maximum number of sources, delete sources every 10 seconds
    if (g_num_sources == MAX_NUM_SOURCES):
        GLib.timeout_add_seconds(10, delete_sources, g_source_bin_list)
        return False

    return True

def restart_source(source_id):
    """Restart a source by stopping and recreating it and its output bin"""
    global g_source_uris, g_source_enabled, g_source_bin_list, g_output_bins, g_rtsp_servers, pipeline

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

    # Wait a moment for cleanup
    time.sleep(0.5)

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

    # Recreate output bin
    output_bin = create_output_bin(source_id)
    if not output_bin:
        print(f"✗ Failed to recreate output for stream {source_id}")
        return False

    g_output_bins[source_id] = output_bin

    # Sync output elements state with pipeline
    for elem in output_bin["elements"]:
        elem.sync_state_with_parent()

    print(f"✓ Restarted source {source_id}")
    return True

def bus_call(bus, message, loop):
    global g_eos_list
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        # Don't quit on source-level errors, only pipeline-level
        src = message.src
        if src and ("source-bin" in src.get_name() or "uridecodebin" in src.get_name()):
            sys.stderr.write(f"Stream source error (non-fatal), continuing...\n")
        else:
            loop.quit()
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
    rtsp_paths = {i: f"/x{i+1}" for i in active}
    return jsonify({"active_streams": active, "count": len(active), "uris": uris, "rtsp_server": "rtsp://localhost:9600", "rtsp_paths": rtsp_paths})

def run_flask():
    app.run(host='0.0.0.0', port=5555, threaded=True)

def main(args):
    global g_num_sources
    global g_source_bin_list
    global g_output_bins
    global g_rtsp_servers
    global uri

    global loop
    global pipeline
    global streammux
    global pgie
    global demux
    global rtsp_server

    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] [uri3] ... \n" % args[0])
        sys.exit(1)

    num_sources=len(args)-1

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streammux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("batch-size", MAX_NUM_SOURCES)
    streammux.set_property("gpu_id", GPU_ID)

    pipeline.add(streammux)
    streammux.set_property("live-source", 1)
    uri = args[1]
    for i in range(num_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=args[i+1]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        #Create first source bin and add to pipeline
        source_bin=create_uridecode_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Failed to create source bin. Exiting. \n")
            sys.exit(1)
        g_source_bin_list[i] = source_bin
        pipeline.add(source_bin)

    g_num_sources = num_sources

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    print("Creating nvstreamdemux \n")
    demux = Gst.ElementFactory.make("nvstreamdemux", "demuxer")
    if not demux:
        sys.stderr.write(" Unable to create nvstreamdemux \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    #Set streammux width and height
    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    #Set pgie configuration file path
    pgie.set_property('config-file-path', PGIE_CONFIG_FILE)

    #Set necessary properties of the nvinfer element, the necessary ones are:
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size < MAX_NUM_SOURCES):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", MAX_NUM_SOURCES," \n")
    pgie.set_property("batch-size",MAX_NUM_SOURCES)

    #Set gpu ID of the inference engine
    pgie.set_property("gpu_id", GPU_ID)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(demux)

    # Link elements: streammux -> pgie -> demux
    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(demux)

    # Create single RTSP server on port 9600
    print("Creating RTSP server on port 9600...\n")
    rtsp_server = GstRtspServer.RTSPServer.new()
    rtsp_server.props.service = "9600"
    rtsp_server.attach(None)
    print("✓ RTSP server created on port 9600\n")

    # Create output bins for each source
    print("Creating output bins for each source...\n")
    for i in range(num_sources):
        output_bin = create_output_bin(i)
        if not output_bin:
            sys.stderr.write(f"Failed to create output bin for source {i}. Exiting.\n")
            sys.exit(1)
        g_output_bins[i] = output_bin

    # Start Flask in separate thread
    print("Starting HTTP control server on port 5555...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    pipeline.set_state(Gst.State.PAUSED)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        if (i != 0):
            print(f"{i}: {source} -> rtsp://localhost:9600/x{i}")

    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)

    print("Waiting for state change\n")
    status, state, pending = pipeline.get_state(Gst.CLOCK_TIME_NONE)
    print(f"Current state {state}, pending state {pending}, status {status}\n")

    # Sync all output elements with pipeline state
    for i in range(num_sources):
        if g_output_bins[i]:
            for elem in g_output_bins[i]["elements"]:
                elem.sync_state_with_parent()

    print("\n" + "="*70)
    print("✓ Individual RTSP output pipeline ready")
    print(f"  Sources: {num_sources}")
    print(f"  Max sources: {MAX_NUM_SOURCES}")
    print(f"  Batch size: {MAX_NUM_SOURCES}")
    print(f"\nRTSP Server: rtsp://localhost:9600")
    print("RTSP Paths:")
    for i in range(num_sources):
        print(f"  Stream {i} (in_s{i}): /x{i+1}")
    print("\nHTTP Control API (port 5555):")
    print("  POST /stream/restart {\"id\": 0}")
    print("  GET  /stream/status")
    print("="*70 + "\n")

    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
