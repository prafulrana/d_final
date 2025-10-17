// Config-driven DeepStream YOLO detection pipeline
// Only custom part: rtspclientsink (push to MediaMTX relay)
#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <string.h>

// Callback when nvurisrcbin creates source pad
typedef struct {
    GstElement *nvstreammux;
    int stream_id;
} PadData;

static void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    PadData *pad_data = (PadData *)data;
    GstPad *sinkpad;
    gchar pad_name[16];

    g_print("[main] New pad '%s' created on nvurisrcbin\n", GST_PAD_NAME(pad));

    snprintf(pad_name, 15, "sink_%u", pad_data->stream_id);
    sinkpad = gst_element_request_pad_simple(pad_data->nvstreammux, pad_name);

    if (gst_pad_link(pad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("[main] Failed to link nvurisrcbin to nvstreammux\n");
    } else {
        g_print("[main] Linked nvurisrcbin:%s -> nvstreammux:%s\n",
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
    GstElement *pipeline, *source, *nvstreammux, *pgie, *nvtracker, *nvvidconv;
    GstElement *nvosd, *rgba_caps, *nvvidconv_postosd, *caps_i420;
    GstElement *encoder, *queue, *h264parse, *rtsp_sink;
    GstBus *bus;
    guint bus_watch_id;
    GstCaps *caps;
    GKeyFile *config;
    GError *error = NULL;

    // Parse arguments
    if (argc != 4) {
        g_printerr("Usage: %s <rtsp_in> <rtsp_out> <config_file>\n", argv[0]);
        return -1;
    }

    const char *rtsp_in = argv[1];
    const char *rtsp_out = argv[2];
    const char *config_file = argv[3];

    g_print("RTSP Input: %s\n", rtsp_in);
    g_print("RTSP Output: %s\n", rtsp_out);
    g_print("Config file: %s\n", config_file);

    // Load config file
    config = g_key_file_new();
    if (!g_key_file_load_from_file(config, config_file, G_KEY_FILE_NONE, &error)) {
        g_printerr("Failed to load config file: %s\n", error->message);
        g_error_free(error);
        return -1;
    }

    // Initialize GStreamer
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);
    pipeline = gst_pipeline_new("deepstream-pipeline");

    // Create elements
    source = gst_element_factory_make("nvurisrcbin", "source");
    nvstreammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie = gst_element_factory_make("nvinfer", "primary-inference");
    nvtracker = gst_element_factory_make("nvtracker", "tracker");
    nvvidconv = gst_element_factory_make("nvvideoconvert", "convertor");
    nvosd = gst_element_factory_make("nvdsosd", "onscreendisplay");
    rgba_caps = gst_element_factory_make("capsfilter", "rgba_caps");
    nvvidconv_postosd = gst_element_factory_make("nvvideoconvert", "convertor-postosd");
    caps_i420 = gst_element_factory_make("capsfilter", "caps_i420");
    encoder = gst_element_factory_make("nvv4l2h264enc", "encoder");
    queue = gst_element_factory_make("queue", "queue");
    h264parse = gst_element_factory_make("h264parse", "h264-parser");
    rtsp_sink = gst_element_factory_make("rtspclientsink", "rtsp-sink");

    if (!pipeline || !source || !nvstreammux || !pgie || !nvtracker || !nvvidconv ||
        !nvosd || !rgba_caps || !nvvidconv_postosd || !caps_i420 ||
        !encoder || !queue || !h264parse || !rtsp_sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    // Configure source from config
    g_object_set(G_OBJECT(source), "uri", rtsp_in, NULL);
    g_object_set(G_OBJECT(source), "rtsp-reconnect-interval", 10, NULL);
    g_object_set(G_OBJECT(source), "init-rtsp-reconnect-interval", 5, NULL);
    g_object_set(G_OBJECT(source), "rtsp-reconnect-attempts", -1, NULL);
    g_object_set(G_OBJECT(source), "select-rtp-protocol", 4, NULL); // TCP

    // Configure streammux from config
    gint mux_width = g_key_file_get_integer(config, "streammux", "width", NULL);
    gint mux_height = g_key_file_get_integer(config, "streammux", "height", NULL);
    gint mux_batch_size = g_key_file_get_integer(config, "streammux", "batch-size", NULL);
    gint mux_timeout = g_key_file_get_integer(config, "streammux", "batched-push-timeout", NULL);
    gint mux_live = g_key_file_get_integer(config, "streammux", "live-source", NULL);

    g_object_set(G_OBJECT(nvstreammux), "width", mux_width, NULL);
    g_object_set(G_OBJECT(nvstreammux), "height", mux_height, NULL);
    g_object_set(G_OBJECT(nvstreammux), "batch-size", mux_batch_size, NULL);
    g_object_set(G_OBJECT(nvstreammux), "batched-push-timeout", mux_timeout, NULL);
    g_object_set(G_OBJECT(nvstreammux), "live-source", mux_live, NULL);

    // Configure inference from config
    gchar *pgie_config_file = g_key_file_get_string(config, "primary-gie", "config-file", NULL);
    g_object_set(G_OBJECT(pgie), "config-file-path", pgie_config_file, NULL);

    // Configure tracker from config
    gchar *tracker_lib = g_key_file_get_string(config, "tracker", "ll-lib-file", NULL);
    gchar *tracker_config_file = g_key_file_get_string(config, "tracker", "ll-config-file", NULL);
    gint tracker_width = g_key_file_get_integer(config, "tracker", "tracker-width", NULL);
    gint tracker_height = g_key_file_get_integer(config, "tracker", "tracker-height", NULL);
    gint tracker_display_id = g_key_file_get_integer(config, "tracker", "display-tracking-id", NULL);

    g_object_set(G_OBJECT(nvtracker), "ll-lib-file", tracker_lib, NULL);
    g_object_set(G_OBJECT(nvtracker), "ll-config-file", tracker_config_file, NULL);
    g_object_set(G_OBJECT(nvtracker), "tracker-width", tracker_width, NULL);
    g_object_set(G_OBJECT(nvtracker), "tracker-height", tracker_height, NULL);
    g_object_set(G_OBJECT(nvtracker), "gpu-id", 0, NULL);
    g_object_set(G_OBJECT(nvtracker), "display-tracking-id", tracker_display_id, NULL);

    // Configure OSD (GPU mode for hardware acceleration)
    g_object_set(G_OBJECT(nvosd), "display-text", 1, NULL);
    g_object_set(G_OBJECT(nvosd), "display-bbox", 1, NULL);
    g_object_set(G_OBJECT(nvosd), "process-mode", 1, NULL);  // GPU mode

    // Configure RGBA capsfilter
    caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA");
    g_object_set(G_OBJECT(rgba_caps), "caps", caps, NULL);
    gst_caps_unref(caps);

    // Configure I420 capsfilter
    caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
    g_object_set(G_OBJECT(caps_i420), "caps", caps, NULL);
    gst_caps_unref(caps);

    // Configure encoder from config
    gint enc_bitrate = g_key_file_get_integer(config, "sink0", "bitrate", NULL);
    gint enc_profile = g_key_file_get_integer(config, "sink0", "profile", NULL);
    gint enc_iframeinterval = g_key_file_get_integer(config, "sink0", "iframeinterval", NULL);

    g_object_set(G_OBJECT(encoder), "bitrate", enc_bitrate, NULL);
    g_object_set(G_OBJECT(encoder), "profile", enc_profile, NULL);
    g_object_set(G_OBJECT(encoder), "insert-sps-pps", 1, NULL);
    g_object_set(G_OBJECT(encoder), "iframeinterval", enc_iframeinterval, NULL);
    g_object_set(G_OBJECT(encoder), "control-rate", 1, NULL);  // CBR
    // Note: preset-level not available in nvv4l2h264enc (DS8)

    // Configure queue (minimal buffering, no frame drops)
    g_object_set(G_OBJECT(queue), "max-size-buffers", 4, NULL);
    g_object_set(G_OBJECT(queue), "max-size-time", 0, NULL);
    g_object_set(G_OBJECT(queue), "max-size-bytes", 0, NULL);

    // Configure h264parse
    g_object_set(G_OBJECT(h264parse), "config-interval", -1, NULL);

    // Configure RTSP sink (custom - push to MediaMTX)
    g_object_set(G_OBJECT(rtsp_sink), "location", rtsp_out, NULL);
    g_object_set(G_OBJECT(rtsp_sink), "protocols", 0x00000004, NULL); // TCP
    g_object_set(G_OBJECT(rtsp_sink), "latency", 100, NULL);  // 100ms for proper timing

    // Add elements to pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, nvstreammux, pgie, nvtracker, nvvidconv,
                     nvosd, rgba_caps, nvvidconv_postosd, caps_i420,
                     encoder, queue, h264parse, rtsp_sink, NULL);

    // Connect pad-added signal for nvurisrcbin (dynamic pads)
    PadData *pad_data = g_new0(PadData, 1);
    pad_data->nvstreammux = nvstreammux;
    pad_data->stream_id = 0;
    g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), pad_data);

    // Link the rest of the pipeline
    if (!gst_element_link_many(nvstreammux, pgie, nvtracker, nvvidconv, nvosd, rgba_caps,
                                nvvidconv_postosd, caps_i420, encoder, queue,
                                h264parse, rtsp_sink, NULL)) {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

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
    g_key_file_free(config);
    g_free(pgie_config_file);
    g_free(tracker_lib);
    g_free(tracker_config_file);

    return 0;
}
