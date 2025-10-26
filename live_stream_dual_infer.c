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
static GHashTable *tracked_objects = NULL;

#define START_ZONE_Y 1200
#define END_ZONE_Y 160

typedef struct {
    gboolean seen_at_start;
    gfloat last_y;
} ObjectState;

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

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);

            // Set OSD colors based on UID: UID 1 (pins) = red, UID 2 (balls) = green
            if (obj_meta->unique_component_id == 1) {
                // Pins: red
                obj_meta->rect_params.border_color.red = 1.0;
                obj_meta->rect_params.border_color.green = 0.0;
                obj_meta->rect_params.border_color.blue = 0.0;
                obj_meta->rect_params.border_color.alpha = 1.0;
            } else if (obj_meta->unique_component_id == 2) {
                // Balls: green
                obj_meta->rect_params.border_color.red = 0.0;
                obj_meta->rect_params.border_color.green = 1.0;
                obj_meta->rect_params.border_color.blue = 0.0;
                obj_meta->rect_params.border_color.alpha = 1.0;
            }

            if (obj_meta->object_id != UNTRACKED_OBJECT_ID) {
                guint64 track_id = obj_meta->object_id;
                gfloat obj_center_y = obj_meta->rect_params.top + (obj_meta->rect_params.height / 2.0);

                ObjectState *state = g_hash_table_lookup(tracked_objects, GUINT_TO_POINTER(track_id));
                if (!state) {
                    state = g_malloc0(sizeof(ObjectState));
                    state->seen_at_start = FALSE;
                    state->last_y = obj_center_y;
                    g_hash_table_insert(tracked_objects, GUINT_TO_POINTER(track_id), state);
                }

                if (obj_center_y > START_ZONE_Y) {
                    state->seen_at_start = TRUE;
                }

                if (state->seen_at_start && obj_center_y < END_ZONE_Y) {
                    throw_counter++;
                    g_print("🎳 THROW #%u detected! (Track ID: %lu)\n", throw_counter, track_id);
                    state->seen_at_start = FALSE;
                }

                state->last_y = obj_center_y;
            }
        }

        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        if (display_meta) {
            NvOSD_TextParams *txt_params = &display_meta->text_params[0];
            display_meta->num_labels = 1;

            txt_params->display_text = g_malloc0(64);
            snprintf(txt_params->display_text, 64, "Throws: %u", throw_counter);

            txt_params->x_offset = 20;
            txt_params->y_offset = 20;

            txt_params->font_params.font_name = "Serif";
            txt_params->font_params.font_size = 24;
            txt_params->font_params.font_color.red = 1.0;
            txt_params->font_params.font_color.green = 1.0;
            txt_params->font_params.font_color.blue = 1.0;
            txt_params->font_params.font_color.alpha = 1.0;

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

    sinkpad = gst_element_request_pad_simple(streammux, "sink_0");

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
    GstElement *pipeline, *source, *streammux, *tee;
    GstElement *queue_pre_pgie1, *queue_pre_pgie2, *queue_post_pgie1, *queue_post_pgie2;
    GstElement *pgie1, *pgie2, *metamux;
    GstElement *nvtracker, *nvvidconv, *nvosd;
    GstElement *nvvidconv_postosd, *caps, *encoder, *queue, *h264parser, *rtppay, *sink;
    GstBus *bus;
    guint bus_watch_id;
    GstCaps *caps_filter;
    GstRTSPServer *server;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;
    GstPad *tee_src0, *tee_src1, *queue_sink0, *queue_sink1;
    GstPad *queue_src0, *queue_src1, *metamux_sink0, *metamux_sink1;

    if (argc < 2 || argc > 4) {
        g_printerr("Usage: %s <stream_id> [orientation] [config_file]\n", argv[0]);
        return -1;
    }

    int stream_id = atoi(argv[1]);

    int is_portrait = 1;
    char pgie1_config[64];
    char pgie2_config[64];
    char tracker_config[64];

    snprintf(pgie1_config, sizeof(pgie1_config), "/config/s%d_pins.txt", stream_id);
    snprintf(pgie2_config, sizeof(pgie2_config), "/config/s%d_ball.txt", stream_id);
    snprintf(tracker_config, sizeof(tracker_config), "/config/s%d_tracker.txt", stream_id);

    if (argc >= 3) {
        if (strcmp(argv[2], "landscape") == 0) {
            is_portrait = 0;
        }
    }

    int width = is_portrait ? 720 : 1920;
    int height = is_portrait ? 1280 : 1080;

    char pipeline_name[32];
    char input_uri[256];
    char rtsp_service[8];
    char rtsp_launch[512];
    char rtsp_url[128];
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

    g_print("\n*** Starting DUAL INFERENCE pipeline for s2 (720x1280) ***\n");
    g_print("*** PGIE1: YOLO12m COCO (UID 1) ***\n");
    g_print("*** PGIE2: Bowling Ball (UID 2) ***\n\n");

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    tracked_objects = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, g_free);

    pipeline = gst_pipeline_new(pipeline_name);

    /* Create all elements */
    source = gst_element_factory_make("nvurisrcbin", "uri-source");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    tee = gst_element_factory_make("tee", "tee");
    queue_pre_pgie1 = gst_element_factory_make("queue", "queue-pre-pgie1");
    queue_pre_pgie2 = gst_element_factory_make("queue", "queue-pre-pgie2");
    pgie1 = gst_element_factory_make("nvinfer", "primary-inference-1");
    pgie2 = gst_element_factory_make("nvinfer", "primary-inference-2");
    queue_post_pgie1 = gst_element_factory_make("queue", "queue-post-pgie1");
    queue_post_pgie2 = gst_element_factory_make("queue", "queue-post-pgie2");
    metamux = gst_element_factory_make("nvdsmetamux", "meta-mux");
    nvtracker = gst_element_factory_make("nvtracker", "tracker");
    nvvidconv = gst_element_factory_make("nvvideoconvert", "convertor");
    nvosd = gst_element_factory_make("nvdsosd", "onscreendisplay");
    nvvidconv_postosd = gst_element_factory_make("nvvideoconvert", "convertor-postosd");
    caps = gst_element_factory_make("capsfilter", "filter");
    encoder = gst_element_factory_make("nvv4l2h264enc", "encoder");
    queue = gst_element_factory_make("queue", "queue");
    h264parser = gst_element_factory_make("h264parse", "h264-parser");
    rtppay = gst_element_factory_make("rtph264pay", "rtppay");
    sink = gst_element_factory_make("udpsink", "udpsink");

    if (!pipeline || !source || !streammux || !tee ||
        !queue_pre_pgie1 || !queue_pre_pgie2 || !queue_post_pgie1 || !queue_post_pgie2 ||
        !pgie1 || !pgie2 || !metamux || !nvtracker || !nvvidconv ||
        !nvosd || !nvvidconv_postosd || !caps || !encoder || !queue ||
        !h264parser || !rtppay || !sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Set properties */
    g_object_set(G_OBJECT(source),
                 "uri", input_uri,
                 "rtsp-reconnect-interval", 10,
                 "rtsp-reconnect-attempts", -1,
                 "latency", 2000,
                 "select-rtp-protocol", 4,
                 NULL);

    g_object_set(G_OBJECT(streammux),
                 "width", width,
                 "height", height,
                 "batch-size", 1,
                 "batched-push-timeout", 33333,
                 "live-source", 0,
                 NULL);

    g_object_set(G_OBJECT(pgie1),
                 "config-file-path", pgie1_config,
                 NULL);

    g_object_set(G_OBJECT(pgie2),
                 "config-file-path", pgie2_config,
                 NULL);

    g_object_set(G_OBJECT(nvtracker),
                 "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                 "ll-config-file", tracker_config,
                 "tracker-width", 1280,
                 "tracker-height", 1280,
                 "gpu-id", 0,
                 "display-tracking-id", 1,
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

    /* Add all elements to pipeline */
    gst_bin_add_many(GST_BIN(pipeline),
                     source, streammux, tee,
                     queue_pre_pgie1, pgie1, queue_post_pgie1,
                     queue_pre_pgie2, pgie2, queue_post_pgie2,
                     metamux, nvtracker, nvvidconv, nvosd,
                     nvvidconv_postosd, caps, encoder, queue,
                     h264parser, rtppay, sink, NULL);

    /* Connect dynamic pad from nvurisrcbin to streammux */
    g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), streammux);

    /* Link main branch: source -> streammux -> tee */
    if (!gst_element_link_many(streammux, tee, NULL)) {
        g_printerr("Failed to link streammux -> tee\n");
        return -1;
    }

    /* Link inference branch 1: queue -> pgie1 -> queue */
    if (!gst_element_link_many(queue_pre_pgie1, pgie1, queue_post_pgie1, NULL)) {
        g_printerr("Failed to link PGIE1 branch\n");
        return -1;
    }

    /* Link inference branch 2: queue -> pgie2 -> queue */
    if (!gst_element_link_many(queue_pre_pgie2, pgie2, queue_post_pgie2, NULL)) {
        g_printerr("Failed to link PGIE2 branch\n");
        return -1;
    }

    /* Link post-metamux: metamux -> tracker -> nvvidconv -> nvosd -> encoder -> sink */
    if (!gst_element_link_many(metamux, nvtracker, nvvidconv, nvosd,
                                nvvidconv_postosd, caps, encoder, queue,
                                h264parser, rtppay, sink, NULL)) {
        g_printerr("Failed to link post-metamux pipeline\n");
        return -1;
    }

    /* Get tee src pads */
    tee_src0 = gst_element_request_pad_simple(tee, "src_0");
    tee_src1 = gst_element_request_pad_simple(tee, "src_1");
    if (!tee_src0 || !tee_src1) {
        g_printerr("Failed to get tee src pads\n");
        return -1;
    }

    /* Get queue sink pads */
    queue_sink0 = gst_element_get_static_pad(queue_pre_pgie1, "sink");
    queue_sink1 = gst_element_get_static_pad(queue_pre_pgie2, "sink");
    if (!queue_sink0 || !queue_sink1) {
        g_printerr("Failed to get queue sink pads\n");
        return -1;
    }

    /* Link tee to queues */
    if (gst_pad_link(tee_src0, queue_sink0) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link tee src_0 to queue_pre_pgie1\n");
        return -1;
    }
    if (gst_pad_link(tee_src1, queue_sink1) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link tee src_1 to queue_pre_pgie2\n");
        return -1;
    }

    gst_object_unref(queue_sink0);
    gst_object_unref(queue_sink1);

    /* Get queue src pads after PGIE */
    queue_src0 = gst_element_get_static_pad(queue_post_pgie1, "src");
    queue_src1 = gst_element_get_static_pad(queue_post_pgie2, "src");
    if (!queue_src0 || !queue_src1) {
        g_printerr("Failed to get queue src pads\n");
        return -1;
    }

    /* Get metamux sink pads */
    metamux_sink0 = gst_element_request_pad_simple(metamux, "sink_0");
    metamux_sink1 = gst_element_request_pad_simple(metamux, "sink_1");
    if (!metamux_sink0 || !metamux_sink1) {
        g_printerr("Failed to get metamux sink pads\n");
        return -1;
    }

    /* Link queues to metamux */
    if (gst_pad_link(queue_src0, metamux_sink0) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link queue_post_pgie1 to metamux sink_0\n");
        return -1;
    }
    if (gst_pad_link(queue_src1, metamux_sink1) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link queue_post_pgie2 to metamux sink_1\n");
        return -1;
    }

    gst_object_unref(queue_src0);
    gst_object_unref(queue_src1);

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
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    /* Add probe to OSD sink pad */
    GstPad *osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    if (!osd_sink_pad) {
        g_print("Unable to get nvosd sink pad\n");
    } else {
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                          osd_sink_pad_buffer_probe, NULL, NULL);
        gst_object_unref(osd_sink_pad);
    }

    /* Start playing */
    g_print("*** Throw counter enabled ***\n\n");
    g_print("*** Pipeline set to PLAYING ***\n\n");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Run main loop */
    g_main_loop_run(loop);

    /* Clean up */
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    if (tracked_objects) {
        g_hash_table_destroy(tracked_objects);
    }

    return 0;
}
