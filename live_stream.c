#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static GMainLoop *loop = NULL;

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n",
                       GST_OBJECT_NAME(msg->src), error->message);
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

static void
on_pad_added(GstElement *element, GstPad *pad, gpointer data)
{
    GstElement *streammux = GST_ELEMENT(data);
    GstPad *sinkpad;

    g_print("Received new pad '%s' from '%s'\n",
            GST_PAD_NAME(pad), GST_ELEMENT_NAME(element));

    sinkpad = gst_element_get_request_pad(streammux, "sink_0");

    if (gst_pad_is_linked(sinkpad)) {
        g_print("Sink pad already linked. Ignoring.\n");
        gst_object_unref(sinkpad);
        return;
    }

    if (gst_pad_link(pad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link pads\n");
    } else {
        g_print("Link succeeded\n");
    }

    gst_object_unref(sinkpad);
}

int main(int argc, char *argv[])
{
    GstElement *pipeline, *source, *streammux, *pgie, *nvvidconv, *nvosd;
    GstElement *nvvidconv_postosd, *caps, *encoder, *queue, *h264parser, *rtppay, *sink;
    GstBus *bus;
    guint bus_watch_id;
    GstCaps *caps_filter;
    GstRTSPServer *server;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;

    /* Parse stream ID from command line */
    if (argc != 2) {
        g_printerr("Usage: %s <stream_id>\n", argv[0]);
        g_printerr("  stream_id: 0, 1, or 2\n");
        return -1;
    }

    int stream_id = atoi(argv[1]);
    if (stream_id < 0 || stream_id > 2) {
        g_printerr("Invalid stream_id: %d. Must be 0, 1, or 2.\n", stream_id);
        return -1;
    }

    /* Derive parameters from stream ID */
    char pipeline_name[32];
    char input_uri[256];
    char rtsp_service[8];
    char rtsp_launch[512];
    char rtsp_url[128];
    char startup_msg[128];
    int udp_port = 5400 + stream_id;
    int rtsp_port = 8554 + stream_id;

    snprintf(pipeline_name, sizeof(pipeline_name), "s%d-pipeline", stream_id);
    snprintf(input_uri, sizeof(input_uri), "rtsp://34.47.221.242:8554/in_s%d", stream_id);
    snprintf(rtsp_service, sizeof(rtsp_service), "%d", rtsp_port);
    snprintf(rtsp_launch, sizeof(rtsp_launch),
             "( udpsrc name=pay0 port=%d buffer-size=524288 "
             "caps=\"application/x-rtp, media=video, clock-rate=90000, "
             "encoding-name=(string)H264, payload=96\" )", udp_port);
    snprintf(rtsp_url, sizeof(rtsp_url), "rtsp://localhost:%d/ds-test", rtsp_port);
    snprintf(startup_msg, sizeof(startup_msg), "Starting pipeline for s%d", stream_id);

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Create pipeline */
    pipeline = gst_pipeline_new(pipeline_name);

    /* Create elements */
    source = gst_element_factory_make("nvurisrcbin", "uri-source");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie = gst_element_factory_make("nvinfer", "primary-inference");
    nvvidconv = gst_element_factory_make("nvvideoconvert", "convertor");
    nvosd = gst_element_factory_make("nvdsosd", "onscreendisplay");
    nvvidconv_postosd = gst_element_factory_make("nvvideoconvert", "convertor-postosd");
    caps = gst_element_factory_make("capsfilter", "filter");
    encoder = gst_element_factory_make("nvv4l2h264enc", "encoder");
    queue = gst_element_factory_make("queue", "queue");
    h264parser = gst_element_factory_make("h264parse", "h264-parser");
    rtppay = gst_element_factory_make("rtph264pay", "rtppay");
    sink = gst_element_factory_make("udpsink", "udpsink");

    if (!pipeline || !source || !streammux || !pgie || !nvvidconv ||
        !nvosd || !nvvidconv_postosd || !caps || !encoder || !queue ||
        !h264parser || !rtppay || !sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Set element properties */
    g_object_set(G_OBJECT(source),
                 "uri", input_uri,
                 "rtsp-reconnect-interval", 10,
                 "rtsp-reconnect-attempts", -1,
                 "latency", 2000,
                 "select-rtp-protocol", 4,  // TCP
                 NULL);

    g_object_set(G_OBJECT(streammux),
                 "width", 1920,
                 "height", 1080,
                 "batch-size", 1,
                 "batched-push-timeout", 33333,
                 "live-source", 0,
                 NULL);

    g_object_set(G_OBJECT(pgie),
                 "config-file-path", "/config/config_infer_yolov8.txt",
                 NULL);

    g_object_set(G_OBJECT(nvosd),
                 "display-text", 1,
                 "display-bbox", 1,
                 "process-mode", 1,
                 "gpu-id", 0,
                 NULL);

    caps_filter = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
    g_object_set(G_OBJECT(caps), "caps", caps_filter, NULL);
    gst_caps_unref(caps_filter);

    g_object_set(G_OBJECT(encoder),
                 "bitrate", 2000000,
                 "iframeinterval", 30,
                 NULL);

    g_object_set(G_OBJECT(h264parser),
                 "config-interval", -1,
                 NULL);

    g_object_set(G_OBJECT(sink),
                 "host", "127.0.0.1",
                 "port", udp_port,
                 "sync", 1,
                 NULL);

    /* Add elements to pipeline */
    gst_bin_add_many(GST_BIN(pipeline),
                     source, streammux, pgie, nvvidconv, nvosd,
                     nvvidconv_postosd, caps, encoder, queue, h264parser, rtppay, sink, NULL);

    /* Connect dynamic pad from nvurisrcbin to streammux */
    g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), streammux);

    /* Link elements */
    if (!gst_element_link_many(streammux, pgie, nvvidconv, nvosd,
                                nvvidconv_postosd, caps, encoder, queue,
                                h264parser, rtppay, sink, NULL)) {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    /* Create RTSP server */
    server = gst_rtsp_server_new();
    g_object_set(server, "service", rtsp_service, NULL);

    mounts = gst_rtsp_server_get_mount_points(server);
    factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, rtsp_launch);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_mount_points_add_factory(mounts, "/ds-test", factory);
    g_object_unref(mounts);
    gst_rtsp_server_attach(server, NULL);
    g_print("\n*** RTSP Server ready at %s ***\n\n", rtsp_url);

    /* Add bus watch */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, NULL);
    gst_object_unref(bus);

    /* Start playing */
    g_print("\n*** %s ***\n\n", startup_msg);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("*** Pipeline set to PLAYING ***\n\n");

    /* Run main loop */
    g_main_loop_run(loop);

    /* Cleanup */
    g_print("Stopping pipeline\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}
