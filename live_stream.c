#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

static GMainLoop *loop = NULL;
static guint throw_counter = 0;
static GHashTable *tracked_objects = NULL;  /* Track object states: Y position */

#define START_ZONE_Y 1200  /* Ball appears here (bottom/throw line) */
#define END_ZONE_Y 160     /* Ball completes throw here (top/pins) */

typedef struct {
    gboolean seen_at_start;
    gfloat last_y;
} ObjectState;

/* Probe callback to count throws and add OSD counter */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        /* Process each tracked object */
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);

            /* Only process objects with valid tracking ID */
            if (obj_meta->object_id != UNTRACKED_OBJECT_ID) {
                guint64 track_id = obj_meta->object_id;
                gfloat obj_center_y = obj_meta->rect_params.top + (obj_meta->rect_params.height / 2.0);

                /* Get or create object state */
                ObjectState *state = g_hash_table_lookup(tracked_objects, GUINT_TO_POINTER(track_id));
                if (!state) {
                    state = g_malloc0(sizeof(ObjectState));
                    state->seen_at_start = FALSE;
                    state->last_y = obj_center_y;
                    g_hash_table_insert(tracked_objects, GUINT_TO_POINTER(track_id), state);
                }

                /* Check if object is in start zone (appears at top) */
                if (obj_center_y > START_ZONE_Y) {
                    state->seen_at_start = TRUE;
                }

                /* Check if object completed journey: start → end (high Y → low Y) */
                if (state->seen_at_start && obj_center_y < END_ZONE_Y) {
                    /* Throw completed! */
                    throw_counter++;
                    g_print("🎳 THROW #%u detected! (Track ID: %lu)\n", throw_counter, track_id);

                    /* Reset state so we can count this track again on next loop */
                    state->seen_at_start = FALSE;
                }

                state->last_y = obj_center_y;
            }
        }

        /* Add custom OSD display for throw counter */
        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        if (display_meta) {
            NvOSD_TextParams *txt_params = &display_meta->text_params[0];
            display_meta->num_labels = 1;

            txt_params->display_text = g_malloc0(64);
            snprintf(txt_params->display_text, 64, "Throws: %u", throw_counter);

            /* Position at top-left */
            txt_params->x_offset = 20;
            txt_params->y_offset = 20;

            /* Font parameters */
            txt_params->font_params.font_name = "Serif";
            txt_params->font_params.font_size = 24;
            txt_params->font_params.font_color.red = 1.0;
            txt_params->font_params.font_color.green = 1.0;
            txt_params->font_params.font_color.blue = 1.0;
            txt_params->font_params.font_color.alpha = 1.0;

            /* Background */
            txt_params->set_bg_clr = 1;
            txt_params->text_bg_clr.red = 0.0;
            txt_params->text_bg_clr.green = 0.0;
            txt_params->text_bg_clr.blue = 0.0;
            txt_params->text_bg_clr.alpha = 0.7;

            nvds_add_display_meta_to_frame(frame_meta, display_meta);
        }
    }

    return GST_PAD_PROBE_OK;
}

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
    GstElement *pipeline, *source, *streammux, *pgie, *sgie, *nvtracker, *nvvidconv, *nvosd;
    GstElement *nvvidconv_postosd, *caps, *encoder, *queue, *h264parser, *rtppay, *sink;
    GstBus *bus;
    guint bus_watch_id;
    GstCaps *caps_filter;
    GstRTSPServer *server;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;
    int enable_tracker;

    /* Parse stream ID, optional orientation, and optional config from command line */
    if (argc < 2 || argc > 4) {
        g_printerr("Usage: %s <stream_id> [orientation] [config_file]\n", argv[0]);
        g_printerr("  stream_id: 0-3\n");
        g_printerr("  orientation: portrait or landscape (default: portrait)\n");
        g_printerr("  config_file: path to inference config (default: /config/config_infer_yolo12x_1280.txt for s0, /config/config_infer_bowling.txt for s3)\n");
        return -1;
    }

    int stream_id = atoi(argv[1]);
    if (stream_id < 0 || stream_id > 3) {
        g_printerr("Invalid stream_id: %d. Must be 0-3.\n", stream_id);
        return -1;
    }

    /* Enable tracker for s1 and s2 only (s3 disabled for testing) */
    enable_tracker = (stream_id == 1 || stream_id == 2);

    /* Disable SGIE for testing */
    int enable_sgie = 0;

    /* Determine orientation: default to portrait for s0-s3 */
    int is_portrait = 1;  /* Default to portrait */
    const char *config_file = NULL;

    if (argc >= 3) {
        if (strcmp(argv[2], "landscape") == 0) {
            is_portrait = 0;
        } else if (strcmp(argv[2], "portrait") == 0) {
            is_portrait = 1;
        } else {
            g_printerr("Invalid orientation: %s. Must be 'portrait' or 'landscape'.\n", argv[2]);
            return -1;
        }
    }

    if (argc == 4) {
        config_file = argv[3];
    } else {
        /* Default config: s0-s1 use COCO, s2 uses 640 bowling, s3 uses 1280 bowling */
        if (stream_id == 2) {
            config_file = "/config/config_infer_bowling_640.txt";
        } else if (stream_id == 3) {
            config_file = "/config/config_infer_bowling.txt";
        } else {
            config_file = "/config/config_infer_yolo12x_1280.txt";
        }
    }

    /* Set dimensions based on orientation */
    int width = is_portrait ? 720 : 1920;
    int height = is_portrait ? 1280 : 1080;

    /* Derive parameters from stream ID */
    char pipeline_name[32];
    char input_uri[256];
    char rtsp_service[8];
    char rtsp_launch[512];
    char rtsp_url[128];
    char startup_msg[256];
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
    if (enable_sgie) {
        snprintf(startup_msg, sizeof(startup_msg), "Starting %s pipeline for s%d (%dx%d) with %s + SGIE pins",
                 is_portrait ? "PORTRAIT" : "LANDSCAPE", stream_id, width, height,
                 strrchr(config_file, '/') ? strrchr(config_file, '/') + 1 : config_file);
    } else {
        snprintf(startup_msg, sizeof(startup_msg), "Starting %s pipeline for s%d (%dx%d) with %s",
                 is_portrait ? "PORTRAIT" : "LANDSCAPE", stream_id, width, height,
                 strrchr(config_file, '/') ? strrchr(config_file, '/') + 1 : config_file);
    }

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Initialize tracked objects hash table for throw counting */
    if (enable_tracker) {
        tracked_objects = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, g_free);
    }

    /* Create pipeline */
    pipeline = gst_pipeline_new(pipeline_name);

    /* Create elements */
    source = gst_element_factory_make("nvurisrcbin", "uri-source");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie = gst_element_factory_make("nvinfer", "primary-inference");
    sgie = enable_sgie ? gst_element_factory_make("nvinfer", "secondary-inference") : NULL;
    nvtracker = enable_tracker ? gst_element_factory_make("nvtracker", "tracker") : NULL;
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
        !h264parser || !rtppay || !sink || (enable_tracker && !nvtracker) ||
        (enable_sgie && !sgie)) {
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
                 "width", width,
                 "height", height,
                 "batch-size", 1,
                 "batched-push-timeout", 33333,
                 "live-source", 0,
                 NULL);

    g_object_set(G_OBJECT(pgie),
                 "config-file-path", config_file,
                 NULL);

    if (enable_sgie) {
        g_object_set(G_OBJECT(sgie),
                     "config-file-path", "/config/s3_bowling_ball_sgie_640.txt",
                     NULL);
    }

    if (enable_tracker) {
        /* s1 uses bowling 640, s3 uses COCO 1280 + ball SGIE */
        const char *tracker_config;
        int tracker_width;
        int tracker_height;

        if (stream_id == 1) {
            tracker_config = "/config/s1_tracker_640.txt";
            tracker_width = 640;
            tracker_height = 640;
        } else if (stream_id == 2) {
            tracker_config = "/config/s2_tracker_640.txt";
            tracker_width = 640;
            tracker_height = 640;
        } else if (stream_id == 3) {
            tracker_config = "/config/s3_tracker_1280.txt";
            tracker_width = 1280;
            tracker_height = 1280;
        } else {
            /* Fallback for s0 if tracker ever enabled */
            tracker_config = "/config/s0_tracker_1280.txt";
            tracker_width = 1280;
            tracker_height = 1280;
        }

        g_object_set(G_OBJECT(nvtracker),
                     "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                     "ll-config-file", tracker_config,
                     "tracker-width", tracker_width,
                     "tracker-height", tracker_height,
                     "gpu-id", 0,
                     "display-tracking-id", 1,
                     NULL);
    }

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
    if (enable_tracker && enable_sgie) {
        gst_bin_add_many(GST_BIN(pipeline),
                         source, streammux, pgie, sgie, nvtracker, nvvidconv, nvosd,
                         nvvidconv_postosd, caps, encoder, queue, h264parser, rtppay, sink, NULL);
    } else if (enable_tracker) {
        gst_bin_add_many(GST_BIN(pipeline),
                         source, streammux, pgie, nvtracker, nvvidconv, nvosd,
                         nvvidconv_postosd, caps, encoder, queue, h264parser, rtppay, sink, NULL);
    } else {
        gst_bin_add_many(GST_BIN(pipeline),
                         source, streammux, pgie, nvvidconv, nvosd,
                         nvvidconv_postosd, caps, encoder, queue, h264parser, rtppay, sink, NULL);
    }

    /* Connect dynamic pad from nvurisrcbin to streammux */
    g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), streammux);

    /* Link elements */
    if (enable_tracker && enable_sgie) {
        if (!gst_element_link_many(streammux, pgie, sgie, nvtracker, nvvidconv, nvosd,
                                    nvvidconv_postosd, caps, encoder, queue,
                                    h264parser, rtppay, sink, NULL)) {
            g_printerr("Elements could not be linked. Exiting.\n");
            return -1;
        }
    } else if (enable_tracker) {
        if (!gst_element_link_many(streammux, pgie, nvtracker, nvvidconv, nvosd,
                                    nvvidconv_postosd, caps, encoder, queue,
                                    h264parser, rtppay, sink, NULL)) {
            g_printerr("Elements could not be linked. Exiting.\n");
            return -1;
        }
    } else {
        if (!gst_element_link_many(streammux, pgie, nvvidconv, nvosd,
                                    nvvidconv_postosd, caps, encoder, queue,
                                    h264parser, rtppay, sink, NULL)) {
            g_printerr("Elements could not be linked. Exiting.\n");
            return -1;
        }
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

    /* Attach probe for throw counter (only if tracking enabled) */
    if (enable_tracker) {
        GstPad *osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
        if (osd_sink_pad) {
            gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                             osd_sink_pad_buffer_probe, NULL, NULL);
            gst_object_unref(osd_sink_pad);
            g_print("*** Throw counter enabled ***\n");
        }
    }

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

    if (tracked_objects) {
        g_hash_table_destroy(tracked_objects);
    }

    return 0;
}
