#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#define NUM_SOURCES 3
static GMainLoop *loop = NULL;

/* Stream info */
typedef struct {
    gint stream_id;        /* 0, 2, 3 */
    gint source_index;     /* 0, 1, 2 for nvstreammux */
    gint rtsp_port;        /* 8554, 8556, 8557 */
} StreamInfo;

StreamInfo streams[NUM_SOURCES] = {
    {0, 0, 8554},  /* s0 */
    {2, 1, 8556},  /* s2 */
    {3, 2, 8557}   /* s3 */
};

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

/* nvmultiurisrcbin outputs directly - no pad-added callback needed! */

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

int
main(int argc, char *argv[])
{
    GstElement *pipeline = NULL;
    GstElement *multi_src = NULL;  /* nvmultiurisrcbin replaces sources+mux */
    GstElement *queue_mux = NULL;
    GstElement *pgie = NULL;
    GstElement *demux = NULL;
    GstElement *cap_head[NUM_SOURCES];
    GstElement *queue[NUM_SOURCES];
    GstElement *vidconv_pre[NUM_SOURCES];
    GstElement *cap_pre[NUM_SOURCES];
    GstElement *nvosd[NUM_SOURCES];
    GstElement *vidconv_post[NUM_SOURCES];
    GstElement *cap_post[NUM_SOURCES];
    GstElement *encoder[NUM_SOURCES];
    GstElement *parser[NUM_SOURCES];
    GstElement *cap_h264[NUM_SOURCES];
    GstElement *payloader[NUM_SOURCES];
    GstElement *queue_sink[NUM_SOURCES];
    GstElement *sink[NUM_SOURCES];

    GstBus *bus = NULL;
    guint bus_watch_id;
    GstRTSPServer *rtsp_server[NUM_SOURCES];
    GstRTSPMountPoints *mounts[NUM_SOURCES];
    GstRTSPMediaFactory *factory[NUM_SOURCES];

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Create pipeline */
    pipeline = gst_pipeline_new("deepstream-multi-pipeline");

    /* Create elements */
    multi_src = gst_element_factory_make("nvmultiurisrcbin", "multi-source");
    queue_mux = gst_element_factory_make("queue", "queue-mux");
    pgie = gst_element_factory_make("nvinfer", "primary-inference");
    demux = gst_element_factory_make("nvstreamdemux", "demuxer");

    if (!pipeline || !multi_src || !queue_mux || !pgie || !demux) {
        g_printerr("Failed to create main elements\n");
        return -1;
    }

    /* Configure nvmultiurisrcbin - REST API mode (port 9000) */
    g_object_set(G_OBJECT(multi_src),
                 "max-batch-size", NUM_SOURCES,
                 "batched-push-timeout", 33000,
                 "width", 720,
                 "height", 1280,
                 "live-source", 1,
                 "cudadec-memtype", 0,
                 "sync-inputs", FALSE,
                 "attach-sys-ts", TRUE,
                 "drop-on-latency", FALSE,  /* Working config uses FALSE */
                 NULL);

    g_print("*** nvmultiurisrcbin REST API server will start on port 9000 ***\n");
    g_print("*** Add streams via: curl -X POST http://localhost:9000/add_stream -d '{\"uri\":\"rtsp://...\", \"id\":0}' ***\n");

    /* Configure PGIE */
    g_object_set(G_OBJECT(pgie),
                 "config-file-path", "/config/config_infer_yolo12x_1280_batch3.txt",
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

        /* Set capsfilter caps - EXACTLY as in working Python code */
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

        /* Configure encoder - from working Python code */
        g_object_set(G_OBJECT(encoder[i]),
                     "insert-sps-pps", 1,
                     "iframeinterval", 30,
                     "bitrate", 3000000,
                     NULL);

        /* Configure queue properties - CRITICAL for preventing busy-wait */
        g_object_set(G_OBJECT(queue[i]),
                     "leaky", 0,
                     "max-size-time", 200000000,  /* 200ms */
                     "max-size-buffers", 0,       /* unlimited */
                     "max-size-bytes", 0,         /* unlimited */
                     NULL);

        g_object_set(G_OBJECT(queue_sink[i]),
                     "leaky", 0,
                     "max-size-time", 200000000,  /* 200ms */
                     "max-size-buffers", 0,       /* unlimited */
                     "max-size-bytes", 0,         /* unlimited */
                     NULL);

        /* Configure rtph264pay */
        g_object_set(G_OBJECT(payloader[i]),
                     "config-interval", -1,  /* Send SPS/PPS with every IDR */
                     "pt", 96,
                     NULL);

        /* Configure UDP sink */
        gint udp_port = 5400 + streams[i].stream_id;
        g_object_set(G_OBJECT(sink[i]),
                     "host", "127.0.0.1",
                     "port", udp_port,
                     "sync", FALSE,
                     "async", FALSE,
                     NULL);

        gst_bin_add_many(GST_BIN(pipeline), cap_head[i], queue[i], vidconv_pre[i], cap_pre[i], nvosd[i],
                        vidconv_post[i], cap_post[i], encoder[i], parser[i], cap_h264[i], payloader[i], queue_sink[i], sink[i], NULL);
    }

    /* Add remaining elements to pipeline */
    gst_bin_add_many(GST_BIN(pipeline), multi_src, queue_mux, pgie, demux, NULL);

    /* Link main pipeline: nvmultiurisrcbin → queue → pgie → demux */
    if (!gst_element_link_many(multi_src, queue_mux, pgie, demux, NULL)) {
        g_printerr("Failed to link main pipeline elements\n");
        return -1;
    }

    /* Link demuxed outputs to per-stream chains */
    for (int i = 0; i < NUM_SOURCES; i++) {
        /* Get demux request pad */
        gchar pad_name[16];
        g_snprintf(pad_name, sizeof(pad_name), "src_%u", i);
        GstPad *demux_src = gst_element_get_request_pad(demux, pad_name);

        if (!demux_src) {
            g_printerr("Failed to get demux pad %s\n", pad_name);
            return -1;
        }

        GstPad *cap_head_sink = gst_element_get_static_pad(cap_head[i], "sink");

        if (gst_pad_link(demux_src, cap_head_sink) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link demux to cap_head %d\n", i);
            return -1;
        }

        gst_object_unref(demux_src);
        gst_object_unref(cap_head_sink);

        /* Link output chain: cap_head → queue → vidconv_pre → cap_pre → nvosd → vidconv_post → cap_post → encoder → parser → cap_h264 → payloader → queue_sink → sink */
        if (!gst_element_link_many(cap_head[i], queue[i], vidconv_pre[i], cap_pre[i], nvosd[i],
                                   vidconv_post[i], cap_post[i], encoder[i], parser[i], cap_h264[i],
                                   payloader[i], queue_sink[i], sink[i], NULL)) {
            g_printerr("Failed to link output chain %d\n", i);
            return -1;
        }
    }

    /* Setup RTSP servers */
    for (int i = 0; i < NUM_SOURCES; i++) {
        rtsp_server[i] = gst_rtsp_server_new();
        g_object_set(G_OBJECT(rtsp_server[i]), "service", g_strdup_printf("%d", streams[i].rtsp_port), NULL);

        mounts[i] = gst_rtsp_server_get_mount_points(rtsp_server[i]);
        factory[i] = gst_rtsp_media_factory_new();

        gchar launch_str[512];
        gint udp_port = 5400 + streams[i].stream_id;
        g_snprintf(launch_str, sizeof(launch_str),
                  "( udpsrc name=pay0 port=%d buffer-size=524288 "
                  "caps=\"application/x-rtp, media=video, clock-rate=90000, "
                  "encoding-name=(string)H264, payload=96\" )", udp_port);

        gst_rtsp_media_factory_set_launch(factory[i], launch_str);
        gst_rtsp_media_factory_set_shared(factory[i], TRUE);
        gst_rtsp_mount_points_add_factory(mounts[i], "/ds-test", factory[i]);
        g_object_unref(mounts[i]);

        gst_rtsp_server_attach(rtsp_server[i], NULL);

        g_print("*** RTSP Server ready at rtsp://localhost:%d/ds-test (s%d) ***\n",
               streams[i].rtsp_port, streams[i].stream_id);
    }

    /* Setup bus */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    g_print("\n*** Starting BATCHED pipeline for s0, s2, s3 with YOLO12x (batch-size=3) ***\n\n");

    /* Start playing */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    g_print("*** Pipeline set to PLAYING ***\n");

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
