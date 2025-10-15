// Pure C++ DeepStream application - zero Python
#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Forward declaration - from segmentation_probe_complete.cpp
extern "C" GstPadProbeReturn segmentation_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);

// Structure to pass streammux to pad-added callback
typedef struct {
    GstElement *nvstreammux;
    int stream_id;
} PadData;

// Callback when nvurisrcbin creates source pad
static void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    PadData *pad_data = (PadData *)data;
    GstPad *sinkpad;
    gchar pad_name[16];

    g_print("[C++] New pad '%s' created on nvurisrcbin\n", GST_PAD_NAME(pad));

    // Request sink pad from nvstreammux
    snprintf(pad_name, 15, "sink_%u", pad_data->stream_id);
    sinkpad = gst_element_request_pad_simple(pad_data->nvstreammux, pad_name);

    if (gst_pad_link(pad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("[C++] Failed to link nvurisrcbin to nvstreammux\n");
    } else {
        g_print("[C++] Linked nvurisrcbin:%s -> nvstreammux:%s\n",
                GST_PAD_NAME(pad), GST_PAD_NAME(sinkpad));
    }

    gst_object_unref(sinkpad);
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *)data;

    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int main(int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline, *source, *nvstreammux, *pgie, *nvvidconv, *nvosd;
    GstElement *rgba_caps, *nvvidconv_postosd, *caps_i420, *encoder, *queue, *h264parse, *rtsp_sink;
    GstBus *bus;
    guint bus_watch_id;
    GstPad *rgba_sinkpad;
    GstCaps *caps;

    // Parse arguments
    if (argc != 4) {
        g_printerr("Usage: %s <rtsp_in> <rtsp_out> <pgie_config>\n", argv[0]);
        return -1;
    }

    const char *rtsp_in = argv[1];
    const char *rtsp_out = argv[2];
    const char *pgie_config = argv[3];

    g_print("RTSP Input: %s\n", rtsp_in);
    g_print("RTSP Output: %s\n", rtsp_out);
    g_print("Inference config: %s\n", pgie_config);

    // Initialize GStreamer
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    // Create pipeline
    pipeline = gst_pipeline_new("deepstream-pipeline");

    // Create elements
    source = gst_element_factory_make("nvurisrcbin", "source");
    nvstreammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie = gst_element_factory_make("nvinfer", "primary-inference");
    nvvidconv = gst_element_factory_make("nvvideoconvert", "convertor");
    nvosd = gst_element_factory_make("nvdsosd", "onscreendisplay");
    rgba_caps = gst_element_factory_make("capsfilter", "rgba_caps");
    nvvidconv_postosd = gst_element_factory_make("nvvideoconvert", "convertor-postosd");
    caps_i420 = gst_element_factory_make("capsfilter", "caps_i420");
    encoder = gst_element_factory_make("nvv4l2h264enc", "encoder");
    queue = gst_element_factory_make("queue", "queue");
    h264parse = gst_element_factory_make("h264parse", "h264-parser");
    rtsp_sink = gst_element_factory_make("rtspclientsink", "rtsp-sink");

    if (!pipeline || !source || !nvstreammux || !pgie || !nvvidconv ||
        !nvosd || !rgba_caps || !nvvidconv_postosd || !caps_i420 ||
        !encoder || !queue || !h264parse || !rtsp_sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    // Configure source
    g_object_set(G_OBJECT(source), "uri", rtsp_in, NULL);
    g_object_set(G_OBJECT(source), "rtsp-reconnect-interval", 10, NULL);
    g_object_set(G_OBJECT(source), "init-rtsp-reconnect-interval", 5, NULL);
    g_object_set(G_OBJECT(source), "rtsp-reconnect-attempts", -1, NULL);
    g_object_set(G_OBJECT(source), "select-rtp-protocol", 4, NULL); // TCP

    // Configure streammux
    g_object_set(G_OBJECT(nvstreammux), "width", 1280, NULL);
    g_object_set(G_OBJECT(nvstreammux), "height", 720, NULL);
    g_object_set(G_OBJECT(nvstreammux), "batch-size", 1, NULL);
    g_object_set(G_OBJECT(nvstreammux), "batched-push-timeout", 4000000, NULL);
    g_object_set(G_OBJECT(nvstreammux), "live-source", 1, NULL);

    // Configure inference
    g_object_set(G_OBJECT(pgie), "config-file-path", pgie_config, NULL);

    // Configure RGBA capsfilter
    caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA");
    g_object_set(G_OBJECT(rgba_caps), "caps", caps, NULL);
    gst_caps_unref(caps);

    // Configure I420 capsfilter
    caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
    g_object_set(G_OBJECT(caps_i420), "caps", caps, NULL);
    gst_caps_unref(caps);

    // Configure encoder
    g_object_set(G_OBJECT(encoder), "bitrate", 3000000, NULL);
    g_object_set(G_OBJECT(encoder), "profile", 2, NULL);
    g_object_set(G_OBJECT(encoder), "insert-sps-pps", 1, NULL);
    g_object_set(G_OBJECT(encoder), "iframeinterval", 30, NULL);

    // Configure h264parse
    g_object_set(G_OBJECT(h264parse), "config-interval", -1, NULL);

    // Configure RTSP sink
    g_object_set(G_OBJECT(rtsp_sink), "location", rtsp_out, NULL);
    g_object_set(G_OBJECT(rtsp_sink), "protocols", 0x00000004, NULL); // TCP
    g_object_set(G_OBJECT(rtsp_sink), "latency", 200, NULL);

    // Add elements to pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, nvstreammux, pgie, nvvidconv,
                     nvosd, rgba_caps, nvvidconv_postosd, caps_i420,
                     encoder, queue, h264parse, rtsp_sink, NULL);

    // Connect pad-added signal for nvurisrcbin (it has dynamic pads)
    PadData *pad_data = g_new0(PadData, 1);
    pad_data->nvstreammux = nvstreammux;
    pad_data->stream_id = 0;
    g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), pad_data);

    // Link the rest of the pipeline
    if (!gst_element_link_many(nvstreammux, pgie, nvvidconv, nvosd, rgba_caps,
                                nvvidconv_postosd, caps_i420, encoder, queue,
                                h264parse, rtsp_sink, NULL)) {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    // Attach probe to rgba_caps sink pad (after nvosd)
    rgba_sinkpad = gst_element_get_static_pad(rgba_caps, "sink");
    if (!rgba_sinkpad) {
        g_printerr("Unable to get sink pad of rgba_caps\n");
        return -1;
    }
    gst_pad_add_probe(rgba_sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
                      segmentation_probe_callback, NULL, NULL);
    g_print("[C++ MAIN] Segmentation probe attached to rgba_caps:sink\n");
    gst_object_unref(rgba_sinkpad);

    // Add bus watch
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    // Start playing
    g_print("Starting pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Wait until error or EOS
    g_print("Running main loop...\n");
    g_main_loop_run(loop);

    // Cleanup
    g_print("Stopping pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}
