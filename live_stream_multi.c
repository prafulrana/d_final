#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#define NUM_SOURCES 4
static GMainLoop *loop = NULL;

/* Stream info */
typedef struct {
    gint stream_id;        /* 0, 1, 2, 3 */
    gint source_index;     /* 0, 1, 2, 3 for nvstreammux */
    gint rtsp_port;        /* 8554, 8555, 8556, 8557 */
} StreamInfo;

StreamInfo streams[NUM_SOURCES] = {
    {0, 0, 8554},  /* s0 */
    {1, 1, 8555},  /* s1 */
    {2, 2, 8556},  /* s2 */
    {3, 3, 8557}   /* s3 */
};

/* Metadata preservation: save UID1 metadata before PGIE2 clears it */
typedef struct {
    guint class_id;
    gchar class_label[64];
    gfloat confidence;
    NvOSD_RectParams rect;
    guint source_id;
} SavedObject;

static GMutex save_mutex;
static GHashTable *saved_metadata = NULL;  /* key: frame_num + source_id, value: GList of SavedObject */

static guint64
make_frame_key(guint source_id, guint64 frame_num)
{
    return ((guint64)source_id << 32) | (frame_num & 0xFFFFFFFF);
}

static GstPadProbeReturn
save_pgie1_metadata(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta) return GST_PAD_PROBE_OK;

    g_mutex_lock(&save_mutex);

    /* Clear old saved metadata */
    if (saved_metadata) {
        GHashTableIter iter;
        gpointer key, value;
        g_hash_table_iter_init(&iter, saved_metadata);
        while (g_hash_table_iter_next(&iter, &key, &value)) {
            g_list_free_full((GList*)value, g_free);
        }
        g_hash_table_destroy(saved_metadata);
    }
    saved_metadata = g_hash_table_new(g_direct_hash, g_direct_equal);

    /* Save all UID1 metadata */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        guint64 key = make_frame_key(frame_meta->source_id, frame_meta->frame_num);
        GList *obj_list = NULL;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)(l_obj->data);
            if (obj->unique_component_id == 1) {  /* UID 1 = pins */
                SavedObject *saved = g_new0(SavedObject, 1);
                saved->class_id = obj->class_id;
                g_strlcpy(saved->class_label, obj->obj_label, sizeof(saved->class_label));
                saved->confidence = obj->confidence;
                saved->rect = obj->rect_params;
                saved->source_id = frame_meta->source_id;
                obj_list = g_list_prepend(obj_list, saved);
            }
        }

        if (obj_list) {
            g_hash_table_insert(saved_metadata, GUINT_TO_POINTER(key), g_list_reverse(obj_list));
        }
    }

    g_mutex_unlock(&save_mutex);
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
merge_pgie2_metadata(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta) return GST_PAD_PROBE_OK;

    g_mutex_lock(&save_mutex);

    if (!saved_metadata) {
        g_mutex_unlock(&save_mutex);
        return GST_PAD_PROBE_OK;
    }

    /* Restore UID1 metadata that PGIE2 cleared */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        guint64 key = make_frame_key(frame_meta->source_id, frame_meta->frame_num);
        GList *saved_objs = g_hash_table_lookup(saved_metadata, GUINT_TO_POINTER(key));

        for (GList *l = saved_objs; l != NULL; l = l->next) {
            SavedObject *saved = (SavedObject *)l->data;

            /* Add saved UID1 object back to frame */
            NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
            obj_meta->unique_component_id = 1;
            obj_meta->class_id = saved->class_id;
            g_strlcpy(obj_meta->obj_label, saved->class_label, MAX_LABEL_SIZE);
            obj_meta->confidence = saved->confidence;
            obj_meta->rect_params = saved->rect;

            nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
        }
    }

    g_mutex_unlock(&save_mutex);
    return GST_PAD_PROBE_OK;
}

/* DEBUG probe to check metadata after inference */
static GstPadProbeReturn
debug_metadata_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    const gchar *probe_name = (const gchar *)u_data;
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta) {
        g_print("[%s] No batch meta!\n", probe_name);
        return GST_PAD_PROBE_OK;
    }

    g_print("\n[%s] Batch has %u frames:\n", probe_name, batch_meta->num_frames_in_batch);

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        guint uid1_count = 0, uid2_count = 0;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            if (obj_meta->unique_component_id == 1) uid1_count++;
            if (obj_meta->unique_component_id == 2) uid2_count++;
        }

        g_print("  Frame src=%u: %u UID1 (pins), %u UID2 (balls), %u total\n",
                frame_meta->source_id, uid1_count, uid2_count, frame_meta->num_obj_meta);
    }

    return GST_PAD_PROBE_OK;
}

/* OSD probe to add stream labels */
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

        /* Add stream ID label */
        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        if (display_meta) {
            NvOSD_TextParams *txt_params = &display_meta->text_params[0];
            display_meta->num_labels = 1;

            gint stream_id = streams[frame_meta->source_id].stream_id;
            txt_params->display_text = g_malloc0(32);
            snprintf(txt_params->display_text, 32, "Stream s%d", stream_id);

            txt_params->x_offset = 20;
            txt_params->y_offset = 20;
            txt_params->font_params.font_name = "Serif";
            txt_params->font_params.font_size = 20;
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

/* Color coding probe - set bbox colors based on UID */
static GstPadProbeReturn
color_code_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
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

            if (obj_meta->unique_component_id == 1) {
                /* UID 1 (pins) = RED */
                obj_meta->rect_params.border_color.red = 1.0;
                obj_meta->rect_params.border_color.green = 0.0;
                obj_meta->rect_params.border_color.blue = 0.0;
                obj_meta->rect_params.border_color.alpha = 1.0;
            } else if (obj_meta->unique_component_id == 2) {
                /* UID 2 (balls) = GREEN */
                obj_meta->rect_params.border_color.red = 0.0;
                obj_meta->rect_params.border_color.green = 1.0;
                obj_meta->rect_params.border_color.blue = 0.0;
                obj_meta->rect_params.border_color.alpha = 1.0;
            }

            obj_meta->rect_params.border_width = 3;
        }
    }

    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
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
        g_printerr("ERROR from element %s: %s\n",
                   GST_OBJECT_NAME(msg->src), error->message);
        if (debug)
            g_printerr("Error details: %s\n", debug);
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

int main(int argc, char *argv[])
{
    GstElement *pipeline;
    GstElement *multi_src, *pgie1, *pgie2, *tracker, *demux;
    GstElement *cap_head[NUM_SOURCES], *queue[NUM_SOURCES];
    GstElement *vidconv_pre[NUM_SOURCES], *cap_pre[NUM_SOURCES];
    GstElement *nvosd[NUM_SOURCES];
    GstElement *vidconv_post[NUM_SOURCES], *cap_post[NUM_SOURCES];
    GstElement *encoder[NUM_SOURCES], *parser[NUM_SOURCES], *cap_h264[NUM_SOURCES];
    GstElement *payloader[NUM_SOURCES], *queue_sink[NUM_SOURCES];
    GstElement *sink[NUM_SOURCES];

    GstBus *bus = NULL;
    guint bus_watch_id;
    GstRTSPServer *rtsp_server[NUM_SOURCES];
    GstRTSPMountPoints *mounts[NUM_SOURCES];
    GstRTSPMediaFactory *factory[NUM_SOURCES];

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);
    g_mutex_init(&save_mutex);

    /* Create pipeline */
    pipeline = gst_pipeline_new("deepstream-multi-pipeline");

    /* Create elements - SEQUENTIAL PGIE ARCHITECTURE */
    multi_src = gst_element_factory_make("nvmultiurisrcbin", "multi-source");
    pgie1 = gst_element_factory_make("nvinfer", "primary-inference-1");
    pgie2 = gst_element_factory_make("nvinfer", "primary-inference-2");
    tracker = gst_element_factory_make("nvtracker", "tracker");
    demux = gst_element_factory_make("nvstreamdemux", "demuxer");

    if (!pipeline || !multi_src || !pgie1 || !pgie2 || !tracker || !demux) {
        g_printerr("Failed to create main elements\n");
        return -1;
    }

    /* Configure nvmultiurisrcbin - Static URIs */
    g_object_set(G_OBJECT(multi_src),
                 "uri-list", "rtsp://34.47.221.242:8554/in_s0,rtsp://34.47.221.242:8554/in_s1,rtsp://34.47.221.242:8554/in_s2,rtsp://34.47.221.242:8554/in_s3",
                 "max-batch-size", NUM_SOURCES,
                 "batched-push-timeout", 33000,
                 "width", 720,
                 "height", 1280,
                 "live-source", 1,
                 "cudadec-memtype", 0,
                 "sync-inputs", FALSE,
                 "attach-sys-ts", TRUE,
                 "drop-on-latency", FALSE,
                 NULL);

    g_print("*** BATCHED DUAL INFERENCE: s0+s1+s2+s3 (YOLO12m pins + bowling ball) ***\n");
    g_print("*** All 4 streams sharing 2 model instances (batch-size=4) ***\n\n");

    /* Configure Dual PGIEs */
    g_object_set(G_OBJECT(pgie1),
                 "config-file-path", "/config/s0_pins_batch4.txt",
                 NULL);

    g_object_set(G_OBJECT(pgie2),
                 "config-file-path", "/config/s0_ball_batch4.txt",
                 NULL);

    /* Configure Tracker */
    g_object_set(G_OBJECT(tracker),
                 "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                 "ll-config-file", "/config/s1_tracker.txt",
                 "tracker-width", 1280,
                 "tracker-height", 1280,
                 "gpu-id", 0,
                 "display-tracking-id", 1,
                 NULL);

    /* Create per-stream output chains */
    for (int i = 0; i < NUM_SOURCES; i++) {
        gchar elem_name[32];

        g_snprintf(elem_name, sizeof(elem_name), "cap-head-%d", i);
        cap_head[i] = gst_element_factory_make("capsfilter", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "queue-%d", i);
        queue[i] = gst_element_factory_make("queue", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "vidconv-pre-%d", i);
        vidconv_pre[i] = gst_element_factory_make("nvvideoconvert", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "cap-pre-%d", i);
        cap_pre[i] = gst_element_factory_make("capsfilter", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "nvosd-%d", i);
        nvosd[i] = gst_element_factory_make("nvdsosd", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "vidconv-post-%d", i);
        vidconv_post[i] = gst_element_factory_make("nvvideoconvert", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "cap-post-%d", i);
        cap_post[i] = gst_element_factory_make("capsfilter", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "encoder-%d", i);
        encoder[i] = gst_element_factory_make("nvv4l2h264enc", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "parser-%d", i);
        parser[i] = gst_element_factory_make("h264parse", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "cap-h264-%d", i);
        cap_h264[i] = gst_element_factory_make("capsfilter", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "payloader-%d", i);
        payloader[i] = gst_element_factory_make("rtph264pay", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "queue-sink-%d", i);
        queue_sink[i] = gst_element_factory_make("queue", elem_name);

        g_snprintf(elem_name, sizeof(elem_name), "udpsink-%d", i);
        sink[i] = gst_element_factory_make("udpsink", elem_name);

        if (!cap_head[i] || !queue[i] || !vidconv_pre[i] || !cap_pre[i] || !nvosd[i] ||
            !vidconv_post[i] || !cap_post[i] || !encoder[i] || !parser[i] || !cap_h264[i] ||
            !payloader[i] || !queue_sink[i] || !sink[i]) {
            g_printerr("Failed to create output elements for stream %d\n", i);
            return -1;
        }

        /* Set capsfilter caps */
        GstCaps *caps_head_nv12 = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
        g_object_set(G_OBJECT(cap_head[i]), "caps", caps_head_nv12, NULL);
        gst_caps_unref(caps_head_nv12);

        GstCaps *caps_rgba = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
        g_object_set(G_OBJECT(cap_pre[i]), "caps", caps_rgba, NULL);
        gst_caps_unref(caps_rgba);

        GstCaps *caps_nv12 = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
        g_object_set(G_OBJECT(cap_post[i]), "caps", caps_nv12, NULL);
        gst_caps_unref(caps_nv12);

        GstCaps *caps_h264_spec = gst_caps_from_string("video/x-h264,stream-format=byte-stream,alignment=au,width=720,height=1280");
        g_object_set(G_OBJECT(cap_h264[i]), "caps", caps_h264_spec, NULL);
        gst_caps_unref(caps_h264_spec);

        /* Configure encoder */
        g_object_set(G_OBJECT(encoder[i]),
                     "insert-sps-pps", 1,
                     "iframeinterval", 30,
                     "bitrate", 3000000,
                     NULL);

        /* Configure queue properties */
        g_object_set(G_OBJECT(queue[i]),
                     "leaky", 0,
                     "max-size-time", 200000000,
                     "max-size-buffers", 0,
                     "max-size-bytes", 0,
                     NULL);

        /* Configure nvosd */
        g_object_set(G_OBJECT(nvosd[i]),
                     "display-text", 1,
                     "display-bbox", 1,
                     "process-mode", 1,
                     "gpu-id", 0,
                     NULL);

        /* Configure udpsink */
        gint udp_port = 5400 + streams[i].stream_id;
        g_object_set(G_OBJECT(sink[i]),
                     "host", "127.0.0.1",
                     "port", udp_port,
                     "sync", 1,
                     NULL);
    }

    /* Add all elements to pipeline */
    gst_bin_add_many(GST_BIN(pipeline),
                     multi_src, pgie1, pgie2, tracker, demux,
                     NULL);

    for (int i = 0; i < NUM_SOURCES; i++) {
        gst_bin_add_many(GST_BIN(pipeline),
                         cap_head[i], queue[i], vidconv_pre[i], cap_pre[i], nvosd[i],
                         vidconv_post[i], cap_post[i], encoder[i], parser[i], cap_h264[i],
                         payloader[i], queue_sink[i], sink[i],
                         NULL);
    }

    /* Link main batch inference chain */
    if (!gst_element_link_many(multi_src, pgie1, pgie2, tracker, demux, NULL)) {
        g_printerr("Failed to link main batch chain\n");
        return -1;
    }

    /* Add metadata probes */
    GstPad *pad_pgie1_src = gst_element_get_static_pad(pgie1, "src");
    gst_pad_add_probe(pad_pgie1_src, GST_PAD_PROBE_TYPE_BUFFER,
                      save_pgie1_metadata, NULL, NULL);
    /* Debug probe removed to reduce CPU usage */
    /* gst_pad_add_probe(pad_pgie1_src, GST_PAD_PROBE_TYPE_BUFFER,
                      debug_metadata_probe, (gpointer)"AFTER_PGIE1", NULL); */
    gst_object_unref(pad_pgie1_src);

    GstPad *pad_pgie2_src = gst_element_get_static_pad(pgie2, "src");
    gst_pad_add_probe(pad_pgie2_src, GST_PAD_PROBE_TYPE_BUFFER,
                      merge_pgie2_metadata, NULL, NULL);
    /* Debug probe removed to reduce CPU usage */
    /* gst_pad_add_probe(pad_pgie2_src, GST_PAD_PROBE_TYPE_BUFFER,
                      debug_metadata_probe, (gpointer)"AFTER_MERGE", NULL); */
    gst_object_unref(pad_pgie2_src);

    /* Link per-stream output chains */
    for (int i = 0; i < NUM_SOURCES; i++) {
        gchar src_pad_name[16];
        g_snprintf(src_pad_name, sizeof(src_pad_name), "src_%u", streams[i].source_index);

        GstPad *demux_src = gst_element_request_pad_simple(demux, src_pad_name);
        GstPad *cap_head_sink = gst_element_get_static_pad(cap_head[i], "sink");

        if (gst_pad_link(demux_src, cap_head_sink) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link demux src_%u to cap_head_%d\n", streams[i].source_index, i);
            return -1;
        }

        gst_object_unref(demux_src);
        gst_object_unref(cap_head_sink);

        if (!gst_element_link_many(cap_head[i], queue[i], vidconv_pre[i], cap_pre[i], nvosd[i],
                                     vidconv_post[i], cap_post[i], encoder[i], parser[i], cap_h264[i],
                                     payloader[i], queue_sink[i], sink[i], NULL)) {
            g_printerr("Failed to link output chain for stream %d\n", i);
            return -1;
        }

        /* Add probes */
        GstPad *nvosd_sink = gst_element_get_static_pad(nvosd[i], "sink");
        gst_pad_add_probe(nvosd_sink, GST_PAD_PROBE_TYPE_BUFFER,
                          osd_sink_pad_buffer_probe, NULL, NULL);
        gst_pad_add_probe(nvosd_sink, GST_PAD_PROBE_TYPE_BUFFER,
                          color_code_buffer_probe, NULL, NULL);
        gst_object_unref(nvosd_sink);
    }

    /* Setup RTSP servers */
    for (int i = 0; i < NUM_SOURCES; i++) {
        rtsp_server[i] = gst_rtsp_server_new();

        gchar service[8];
        g_snprintf(service, sizeof(service), "%d", streams[i].rtsp_port);
        g_object_set(rtsp_server[i], "service", service, NULL);

        mounts[i] = gst_rtsp_server_get_mount_points(rtsp_server[i]);
        factory[i] = gst_rtsp_media_factory_new();

        gint udp_port = 5400 + streams[i].stream_id;
        gchar rtsp_launch[512];
        g_snprintf(rtsp_launch, sizeof(rtsp_launch),
                   "( udpsrc name=pay0 port=%d buffer-size=524288 "
                   "caps=\"application/x-rtp, media=video, clock-rate=90000, "
                   "encoding-name=(string)H264, payload=96\" )", udp_port);

        gst_rtsp_media_factory_set_launch(factory[i], rtsp_launch);
        gst_rtsp_media_factory_set_shared(factory[i], TRUE);
        gst_rtsp_mount_points_add_factory(mounts[i], "/ds-test", factory[i]);
        g_object_unref(mounts[i]);
        gst_rtsp_server_attach(rtsp_server[i], NULL);

        g_print("*** RTSP s%d ready at rtsp://localhost:%d/ds-test ***\n",
                streams[i].stream_id, streams[i].rtsp_port);
    }

    /* Set up message bus */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    /* Start pipeline */
    g_print("\n*** Starting pipeline ***\n\n");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Run main loop */
    g_main_loop_run(loop);

    /* Clean up */
    g_print("Stopping pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    if (saved_metadata) {
        GHashTableIter iter;
        gpointer key, value;
        g_hash_table_iter_init(&iter, saved_metadata);
        while (g_hash_table_iter_next(&iter, &key, &value)) {
            g_list_free_full((GList*)value, g_free);
        }
        g_hash_table_destroy(saved_metadata);
    }
    g_mutex_clear(&save_mutex);

    return 0;
}
