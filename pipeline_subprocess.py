#!/usr/bin/env python3
"""DeepStream pipeline running in isolated subprocess"""

import sys
import time
import multiprocessing
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

from config import *

# Global state (within subprocess)
g_num_sources = 0
g_eos_list = [False] * MAX_NUM_SOURCES
g_source_enabled = [False] * MAX_NUM_SOURCES
g_source_bin_list = [None] * MAX_NUM_SOURCES
g_output_bins = [None] * MAX_NUM_SOURCES
g_relay_host = None

# Pipeline elements
loop = None
pipeline = None
streammux = None
pgie = None
demux = None
rtsp_server = None

# Communication queue
command_queue = None
response_queue = None


def get_source_uri(source_id):
    """Generate RTSP input URI from relay host and source ID"""
    return f"rtsp://{g_relay_host}:8554/in_s{source_id}"


def decodebin_child_added(child_proxy, Object, name, user_data):
    """Handle decodebin child elements - set GPU ID and force TCP for RTSP"""
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    elif name.find("nvv4l2decoder") != -1:
        Object.set_property("gpu_id", GPU_ID)
    elif name.find("source") != -1:
        try:
            Object.set_property("protocols", 0x00000004)  # TCP
            print(f"✓ Set {name} to TCP-only")
        except Exception:
            pass


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


def create_uridecode_bin(index):
    """Create uridecodebin for a given source"""
    global g_source_enabled

    uri = get_source_uri(index)
    print(f"Creating uridecodebin for [{uri}]")

    bin_name = f"source-bin-{index:02d}"
    bin = Gst.ElementFactory.make("uridecodebin", bin_name)
    if not bin:
        sys.stderr.write(f"Unable to create uri decode bin {bin_name}\n")
        return None

    bin.set_property("uri", uri)
    bin.connect("pad-added", cb_newpad, index)
    bin.connect("child-added", decodebin_child_added, index)
    g_source_enabled[index] = True

    return bin


def create_output_bin(source_id):
    """Create output chain for source"""
    global pipeline, demux, rtsp_server

    pad_name = f"src_{source_id}"
    demux_src = demux.request_pad_simple(pad_name)
    if not demux_src:
        print(f"✗ Failed to get demux pad {pad_name}")
        return None

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
    encoder.set_property("bitrate", ENCODER_BITRATE)
    encoder.set_property("insert-sps-pps", True)
    encoder.set_property("idrinterval", ENCODER_IDR_INTERVAL)
    encoder.set_property("iframeinterval", ENCODER_IFRAME_INTERVAL)

    rtppay = Gst.ElementFactory.make("rtph264pay", f"rtppay-{source_id}")
    udpsink = Gst.ElementFactory.make("udpsink", f"udpsink-{source_id}")
    udpsink.set_property("host", "127.0.0.1")
    udpsink.set_property("port", RTSP_UDPSINK_BASE_PORT + source_id)
    udpsink.set_property("async", False)
    udpsink.set_property("sync", 0)

    elements = [queue, nvvidconv, caps_filter, nvosd, nvvidconv2, caps_filter2, encoder, rtppay, udpsink]

    for elem in elements:
        pipeline.add(elem)

    queue_sink = queue.get_static_pad("sink")
    if demux_src.link(queue_sink) != Gst.PadLinkReturn.OK:
        print(f"✗ Failed to link demux to queue for stream {source_id}")
        return None

    if not queue.link(nvvidconv):
        return None
    if not nvvidconv.link(caps_filter):
        return None
    if not caps_filter.link(nvosd):
        return None
    if not nvosd.link(nvvidconv2):
        return None
    if not nvvidconv2.link(caps_filter2):
        return None
    if not caps_filter2.link(encoder):
        return None
    if not encoder.link(rtppay):
        return None
    if not rtppay.link(udpsink):
        return None

    return {"elements": elements, "demux_src": demux_src}


def stop_release_source(source_id):
    """Stop and release a source bin from the pipeline"""
    global g_num_sources, g_source_bin_list, g_output_bins, g_source_enabled, streammux, demux, pipeline

    state_return = g_source_bin_list[source_id].set_state(Gst.State.NULL)

    if state_return == Gst.StateChangeReturn.ASYNC:
        # Wait for state change to complete
        state_return = g_source_bin_list[source_id].get_state(Gst.CLOCK_TIME_NONE)[0]

    if state_return != Gst.StateChangeReturn.FAILURE:
        pad_name = f"sink_{source_id}"
        sinkpad = streammux.get_static_pad(pad_name)
        if sinkpad:
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            streammux.release_request_pad(sinkpad)

        pipeline.remove(g_source_bin_list[source_id])
        g_source_bin_list[source_id] = None
        g_source_enabled[source_id] = False
        g_num_sources -= 1
        print(f"Released source {source_id}")
    else:
        print(f"Failed to stop source {source_id}")

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

        # Clean up output bin elements
        try:
            if g_output_bins[source_id].get("elements"):
                for elem in g_output_bins[source_id]["elements"]:
                    elem.set_state(Gst.State.NULL)
                    pipeline.remove(elem)
        except Exception as e:
            print(f"Warning during output bin cleanup: {e}")

        # Clear output bin reference
        g_output_bins[source_id] = None

    # Clear EOS flag
    g_eos_list[source_id] = False


def restart_source(source_id):
    """Restart a source by stopping and recreating it"""
    global g_source_enabled, g_source_bin_list, g_output_bins, pipeline, g_num_sources, rtsp_server

    if source_id < 0 or source_id >= MAX_NUM_SOURCES:
        print(f"Invalid source_id {source_id}")
        return False

    uri = get_source_uri(source_id)
    print(f"Restarting source {source_id}: {uri}")

    if g_source_enabled[source_id]:
        stop_release_source(source_id)
        time.sleep(SOURCE_RESTART_DELAY_SEC)

    source_bin = create_uridecode_bin(source_id)
    if not source_bin:
        print(f"Failed to recreate source {source_id}")
        return False

    g_source_bin_list[source_id] = source_bin
    pipeline.add(source_bin)

    state_return = source_bin.set_state(Gst.State.PLAYING)
    if state_return == Gst.StateChangeReturn.ASYNC:
        source_bin.get_state(Gst.CLOCK_TIME_NONE)

    if not g_source_enabled[source_id]:
        g_source_enabled[source_id] = True
        g_num_sources += 1
        print(f"✓ Enabled source {source_id}")

    if g_output_bins[source_id] and g_output_bins[source_id].get("elements"):
        try:
            pad_name = f"src_{source_id}"
            demux_src = demux.request_pad_simple(pad_name)
            if not demux_src:
                print(f"✗ Failed to get demux pad {pad_name}")
                return False

            queue = g_output_bins[source_id]["elements"][0]
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
        output_bin = create_output_bin(source_id)
        if not output_bin:
            print(f"✗ Failed to create output for stream {source_id}")
            return False

        g_output_bins[source_id] = output_bin

        for elem in output_bin["elements"]:
            state_return = elem.set_state(Gst.State.PLAYING)
            if state_return == Gst.StateChangeReturn.ASYNC:
                elem.get_state(Gst.CLOCK_TIME_NONE)

        try:
            mounts = rtsp_server.get_mount_points()
            path = f"/x{source_id}"
            port = RTSP_UDPSINK_BASE_PORT + source_id
            factory = GstRtspServer.RTSPMediaFactory.new()
            launch_str = f"( udpsrc name=pay0 port={port} buffer-size={RTSP_UDP_BUFFER_SIZE} caps=\"{RTSP_CAPS_STRING}\" )"
            factory.set_launch(launch_str)
            factory.set_shared(True)
            mounts.add_factory(path, factory)
            print(f"✓ RTSP factory created at {path} (listening on UDP {port})")
        except Exception as e:
            print(f"⚠ Failed to create RTSP factory for {path}: {e}")

    print(f"✓ Restarted source {source_id}")
    return True


def bus_call(bus, message, user_data):
    """Handle GStreamer bus messages"""
    global g_eos_list, loop
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream - all sources EOS'd\n")
        print("\n" + "="*70)
        print("All sources ended - shutting down subprocess")
        print("="*70 + "\n")
        response_queue.put({"event": "EOS"})
        if loop:
            loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write(f"Warning: {err}: {debug}\n")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"Error: {err}: {debug}\n")
    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        if struct is not None and struct.has_name("stream-eos"):
            parsed, stream_id = struct.get_uint("stream-id")
            if parsed:
                print(f"Got EOS from stream {stream_id}")
                g_eos_list[stream_id] = True
    return True


def create_pipeline(relay_host):
    """Create and start the DeepStream pipeline"""
    global g_num_sources, g_source_bin_list, g_output_bins, g_relay_host
    global loop, pipeline, streammux, pgie, demux, rtsp_server

    Gst.init(None)
    g_relay_host = relay_host

    print("Creating pipeline")
    pipeline = Gst.Pipeline()
    if not pipeline:
        return False

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        return False

    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("batch-size", MAX_NUM_SOURCES)
    streammux.set_property("gpu_id", GPU_ID)
    streammux.set_property("live-source", 1)
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)

    pipeline.add(streammux)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        return False
    pgie.set_property("config-file-path", PGIE_CONFIG_FILE)
    pgie.set_property("batch-size", MAX_NUM_SOURCES)
    pgie.set_property("gpu_id", GPU_ID)

    demux = Gst.ElementFactory.make("nvstreamdemux", "demuxer")
    if not demux:
        return False

    pipeline.add(pgie)
    pipeline.add(demux)

    streammux.link(pgie)
    pgie.link(demux)

    rtsp_server = GstRtspServer.RTSPServer.new()
    rtsp_server.props.service = str(RTSP_SERVER_PORT)
    rtsp_server.attach(None)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, None)

    pipeline.set_state(Gst.State.PLAYING)

    print("✓ Pipeline ready (subprocess)")
    return True


def get_status():
    """Get current pipeline status"""
    active = [i for i in range(MAX_NUM_SOURCES) if g_source_enabled[i]]
    uris = {i: get_source_uri(i) for i in active}
    rtsp_paths = {i: f"/x{i}" for i in active}
    return {
        "active_streams": active,
        "count": len(active),
        "uris": uris,
        "rtsp_server": f"rtsp://localhost:{RTSP_SERVER_PORT}",
        "rtsp_paths": rtsp_paths
    }


def handle_start_source(source_id):
    """Handle START_SOURCE command in GLib main thread"""
    success = restart_source(source_id)
    response_queue.put({"status": "ok" if success else "error", "id": source_id})
    return False  # Don't repeat


def handle_get_status():
    """Handle GET_STATUS command in GLib main thread"""
    status = get_status()
    response_queue.put(status)
    return False  # Don't repeat


def handle_shutdown():
    """Handle SHUTDOWN command in GLib main thread"""
    global loop
    print("Shutdown command received")
    if loop:
        loop.quit()
    response_queue.put({"status": "shutdown"})
    return False  # Don't repeat


def command_thread():
    """Background thread that waits for commands and dispatches to GLib main thread"""
    while True:
        try:
            cmd = command_queue.get()  # BLOCKING - only wakes when command arrives

            if cmd["cmd"] == "START_SOURCE":
                source_id = cmd["id"]
                GLib.idle_add(handle_start_source, source_id)

            elif cmd["cmd"] == "GET_STATUS":
                GLib.idle_add(handle_get_status)

            elif cmd["cmd"] == "SHUTDOWN":
                GLib.idle_add(handle_shutdown)
                break  # Exit thread after shutdown

        except Exception as e:
            print(f"Command thread error: {e}")


def subprocess_main(relay_host, cmd_queue, resp_queue):
    """Main entry point for subprocess"""
    import threading
    global command_queue, response_queue
    command_queue = cmd_queue
    response_queue = resp_queue

    print(f"Pipeline subprocess starting (relay: {relay_host})")

    if not create_pipeline(relay_host):
        print("Failed to create pipeline")
        response_queue.put({"event": "INIT_FAILED"})
        return

    response_queue.put({"event": "INIT_SUCCESS"})

    # Start command handler in background thread
    cmd_thread = threading.Thread(target=command_thread, daemon=True)
    cmd_thread.start()

    # Run GLib loop (blocking, processes pipeline frames)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted")

    print("Pipeline subprocess exiting")
