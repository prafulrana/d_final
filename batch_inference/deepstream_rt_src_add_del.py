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
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
GPU_ID = 0
MAX_NUM_SOURCES = 36
PGIE_CONFIG_FILE = "/config/bowling_yolo12n_batch.txt"
TRACKER_CONFIG_FILE = "config_tracker_NvDCF_perf.yml"
RTSP_OUTPUT_PORT = 8555

CONFIG_GPU_ID = "gpu-id"
CONFIG_GROUP_TRACKER = "tracker"
CONFIG_GROUP_TRACKER_WIDTH = "tracker-width"
CONFIG_GROUP_TRACKER_HEIGHT = "tracker-height"
CONFIG_GROUP_TRACKER_LL_CONFIG_FILE = "ll-config-file"
CONFIG_GROUP_TRACKER_LL_LIB_FILE = "ll-lib-file"

g_num_sources = 0
g_source_id_list = [0] * MAX_NUM_SOURCES
g_eos_list = [False] * MAX_NUM_SOURCES
g_source_enabled = [False] * MAX_NUM_SOURCES
g_source_bin_list = [None] * MAX_NUM_SOURCES
g_source_uris = {}  # Store original URIs for each source

pgie_classes_str= ["bowling-ball", "bowling-pins", "sweep-board"]

uri = ""

loop = None
pipeline = None
streammux = None
pgie = None
nvvideoconvert = None
nvosd = None
tiler = None
tracker = None
rtsp_server = None

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


def stop_release_source(source_id):
    global g_num_sources
    global g_source_bin_list
    global streammux
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
        source_id -= 1
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
        source_id -= 1
        g_num_sources -= 1


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
    """Restart a source by stopping and recreating it"""
    global g_source_uris, g_source_enabled, g_source_bin_list, pipeline

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
    return jsonify({"active_streams": active, "count": len(active), "uris": uris})

def run_flask():
    app.run(host='0.0.0.0', port=5555, threaded=True)

def main(args):
    global g_num_sources
    global g_source_bin_list
    global uri

    global loop
    global pipeline
    global streammux
    global pgie
    global nvvideoconvert
    global nvosd
    global tiler
    global tracker
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

    streammux.set_property("batched-push-timeout", 25000)
    streammux.set_property("batch-size", 30)
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

    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    print("Creating nvvidconv \n ")
    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvideoconvert:
        sys.stderr.write(" Unable to create nvvidconv \n")

    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    print("Creating nvvidconv2 \n")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv2")
    if not nvvidconv2:
        sys.stderr.write(" Unable to create nvvidconv2 \n")

    print("Creating capsfilter \n")
    caps_filter = Gst.ElementFactory.make("capsfilter", "caps")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    caps_filter.set_property("caps", caps)

    print("Creating encoder \n")
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder \n")
    encoder.set_property("bitrate", 4000000)

    print("Creating rtppay \n")
    rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay \n")

    print("Creating udpsink \n")
    udpsink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not udpsink:
        sys.stderr.write(" Unable to create udpsink \n")
    udpsink.set_property("host", "127.0.0.1")
    udpsink.set_property("port", 5000)
    udpsink.set_property("async", False)
    udpsink.set_property("sync", 0)
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
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", num_sources," \n")
    pgie.set_property("batch-size",MAX_NUM_SOURCES)

    #Set gpu ID of the inference engine
    pgie.set_property("gpu_id", GPU_ID)

    #Set tiler properties
    tiler_rows=int(math.sqrt(num_sources))
    tiler_columns=int(math.ceil((1.0*num_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    #Set gpu IDs
    tiler.set_property("gpu_id", GPU_ID)
    nvvideoconvert.set_property("gpu_id", GPU_ID)
    nvosd.set_property("gpu_id", GPU_ID)
    nvvidconv2.set_property("gpu_id", GPU_ID)
    encoder.set_property("gpu-id", GPU_ID)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvideoconvert)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv2)
    pipeline.add(caps_filter)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(udpsink)

    # Link elements: streammux -> pgie -> tiler -> nvvidconv -> nvosd -> nvvidconv2 -> caps -> encoder -> rtppay -> udpsink
    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(tiler)
    tiler.link(nvvideoconvert)
    nvvideoconvert.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(caps_filter)
    caps_filter.link(encoder)
    encoder.link(rtppay)
    rtppay.link(udpsink)

    udpsink.set_property("sync", 0)
    udpsink.set_property("qos", 0)

    # Create RTSP server
    print("Creating RTSP server \n")
    rtsp_server = GstRtspServer.RTSPServer.new()
    rtsp_server.props.service = str(RTSP_OUTPUT_PORT)

    factory = GstRtspServer.RTSPMediaFactory.new()
    launch_str = f"( udpsrc name=pay0 port=5000 buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96\" )"
    factory.set_launch(launch_str)
    factory.set_shared(True)

    mounts = rtsp_server.get_mount_points()
    mounts.add_factory("/ds-test", factory)
    rtsp_server.attach(None)

    print(f"✓ RTSP server ready at rtsp://localhost:{RTSP_OUTPUT_PORT}/ds-test")

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
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)

    print("Waiting for state change\n")
    status, state, pending = pipeline.get_state(Gst.CLOCK_TIME_NONE)
    print(f"Current state {state}, pending state {pending}, status {status}\n")
    # Disabled auto-add to prevent duplicate sources
    # if (state == Gst.State.PLAYING) and (pending == Gst.State.VOID_PENDING):
    #     GLib.timeout_add_seconds(10, add_sources, g_source_bin_list)
    # else:
    #     print("Pipeline is not in playing state to add sources\n")

    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
