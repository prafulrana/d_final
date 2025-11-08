#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <pthread.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#define NUM_SOURCES 2
#define API_PORT 9000

static GMainLoop *loop = NULL;
static GstElement *pipeline = NULL;
static GstElement *streammux = NULL;
static GstElement *demux = NULL;
static gint active_source_count = 0;

/* Stream info */
typedef struct {
    gint stream_id;        /* 0-35 */
    gint source_index;     /* 0-35 for nvstreammux */
    gint rtsp_port;        /* 8554-8589 */
    gboolean active;       /* Is this stream slot active? */
    GstElement *source;    /* nvurisrcbin element */
} StreamInfo;

StreamInfo streams[NUM_SOURCES] = {
    {0, 0, 8554, FALSE, NULL}, {1, 1, 8555, FALSE, NULL}, {2, 2, 8556, FALSE, NULL}, {3, 3, 8557, FALSE, NULL},
    {4, 4, 8558, FALSE, NULL}, {5, 5, 8559, FALSE, NULL}, {6, 6, 8560, FALSE, NULL}, {7, 7, 8561, FALSE, NULL},
    {8, 8, 8562, FALSE, NULL}, {9, 9, 8563, FALSE, NULL}, {10, 10, 8564, FALSE, NULL}, {11, 11, 8565, FALSE, NULL},
    {12, 12, 8566, FALSE, NULL}, {13, 13, 8567, FALSE, NULL}, {14, 14, 8568, FALSE, NULL}, {15, 15, 8569, FALSE, NULL},
    {16, 16, 8570, FALSE, NULL}, {17, 17, 8571, FALSE, NULL}, {18, 18, 8572, FALSE, NULL}, {19, 19, 8573, FALSE, NULL},
    {20, 20, 8574, FALSE, NULL}, {21, 21, 8575, FALSE, NULL}, {22, 22, 8576, FALSE, NULL}, {23, 23, 8577, FALSE, NULL},
    {24, 24, 8578, FALSE, NULL}, {25, 25, 8579, FALSE, NULL}, {26, 26, 8580, FALSE, NULL}, {27, 27, 8581, FALSE, NULL},
    {28, 28, 8582, FALSE, NULL}, {29, 29, 8583, FALSE, NULL}, {30, 30, 8584, FALSE, NULL}, {31, 31, 8585, FALSE, NULL},
    {32, 32, 8586, FALSE, NULL}, {33, 33, 8587, FALSE, NULL}, {34, 34, 8588, FALSE, NULL}, {35, 35, 8589, FALSE, NULL}
};

/* Output chain elements (pre-created) */
static GstElement *cap_head[NUM_SOURCES];
static GstElement *queue[NUM_SOURCES];
static GstElement *vidconv_pre[NUM_SOURCES];
static GstElement *cap_pre[NUM_SOURCES];
static GstElement *nvosd[NUM_SOURCES];
static GstElement *vidconv_post[NUM_SOURCES];
static GstElement *cap_post[NUM_SOURCES];
static GstElement *encoder[NUM_SOURCES];
static GstElement *parser[NUM_SOURCES];
static GstElement *cap_h264[NUM_SOURCES];
static GstElement *payloader[NUM_SOURCES];
static GstElement *queue_sink[NUM_SOURCES];
static GstElement *sink[NUM_SOURCES];

static GMutex source_lock;

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

    case GST_MESSAGE_STATE_CHANGED: {
        if (GST_MESSAGE_SRC(msg) == GST_OBJECT(pipeline)) {
            GstState old_state, new_state;
            gst_message_parse_state_changed(msg, &old_state, &new_state, NULL);
            if (new_state == GST_STATE_PLAYING) {
                g_print("\n*** Pipeline set to PLAYING - Ready for sources ***\n\n");
            }
        }
        break;
    }

    default:
        break;
    }

    return TRUE;
}

/* Callback for dynamic pad linking from nvurisrcbin to streammux */
static void
on_pad_added(GstElement *element, GstPad *pad, gpointer data)
{
    gint stream_id = GPOINTER_TO_INT(data);
    gchar pad_name[16];
    GstPad *sinkpad = NULL;

    g_snprintf(pad_name, sizeof(pad_name), "sink_%d", stream_id);

    g_print("Received new pad '%s' from 'source-%d', linking to %s\n",
            GST_PAD_NAME(pad), stream_id, pad_name);

    sinkpad = gst_element_request_pad_simple(streammux, pad_name);
    if (!sinkpad) {
        g_printerr("Failed to get sink pad %s\n", pad_name);
        return;
    }

    if (gst_pad_is_linked(sinkpad)) {
        g_print("Sink pad %s already linked. Ignoring.\n", pad_name);
        gst_object_unref(sinkpad);
        return;
    }

    if (gst_pad_link(pad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link pads\n");
    } else {
        g_print("✓ Stream s%d linked successfully\n", stream_id);
    }

    gst_object_unref(sinkpad);
}

/* Add source dynamically - called from GLib main loop context */
static gboolean
add_source_idle(gpointer data)
{
    gint stream_id = GPOINTER_TO_INT(data);

    g_mutex_lock(&source_lock);

    if (stream_id < 0 || stream_id >= NUM_SOURCES) {
        g_printerr("Invalid stream_id: %d\n", stream_id);
        g_mutex_unlock(&source_lock);
        return FALSE;
    }

    if (streams[stream_id].active) {
        g_printerr("Stream s%d already active\n", stream_id);
        g_mutex_unlock(&source_lock);
        return FALSE;
    }

    g_print("Adding source s%d...\n", stream_id);

    /* Create nvurisrcbin */
    gchar elem_name[32];
    g_snprintf(elem_name, sizeof(elem_name), "source-%d", stream_id);
    streams[stream_id].source = gst_element_factory_make("nvurisrcbin", elem_name);

    if (!streams[stream_id].source) {
        g_printerr("Failed to create source-%d\n", stream_id);
        g_mutex_unlock(&source_lock);
        return FALSE;
    }

    /* Set source URI */
    gchar input_uri[128];
    g_snprintf(input_uri, sizeof(input_uri), "rtsp://34.47.221.242:8554/in_s%d", stream_id);

    g_object_set(G_OBJECT(streams[stream_id].source),
                 "uri", input_uri,
                 "rtsp-reconnect-interval", 10,
                 "rtsp-reconnect-attempts", -1,
                 "latency", 2000,
                 "select-rtp-protocol", 4,
                 NULL);

    /* Connect pad-added signal */
    g_signal_connect(streams[stream_id].source, "pad-added",
                     G_CALLBACK(on_pad_added), GINT_TO_POINTER(stream_id));

    /* Add to pipeline and sync with parent state (already PLAYING) */
    gst_bin_add(GST_BIN(pipeline), streams[stream_id].source);
    gst_element_sync_state_with_parent(streams[stream_id].source);

    streams[stream_id].active = TRUE;
    active_source_count++;
    g_print("✓ Stream s%d added (rtsp://34.47.221.242:8554/in_s%d)\n", stream_id, stream_id);

    g_mutex_unlock(&source_lock);
    return FALSE;
}

/* Delete source dynamically - called from GLib main loop context */
static gboolean
delete_source_idle(gpointer data)
{
    gint stream_id = GPOINTER_TO_INT(data);

    g_mutex_lock(&source_lock);

    if (stream_id < 0 || stream_id >= NUM_SOURCES) {
        g_printerr("Invalid stream_id: %d\n", stream_id);
        g_mutex_unlock(&source_lock);
        return FALSE;
    }

    if (!streams[stream_id].active) {
        g_printerr("Stream s%d not active\n", stream_id);
        g_mutex_unlock(&source_lock);
        return FALSE;
    }

    g_print("Deleting source s%d...\n", stream_id);

    /* Set source to NULL and remove from pipeline */
    gst_element_set_state(streams[stream_id].source, GST_STATE_NULL);

    /* Release streammux pad */
    gchar pad_name[16];
    g_snprintf(pad_name, sizeof(pad_name), "sink_%d", stream_id);
    GstPad *sinkpad = gst_element_get_static_pad(streammux, pad_name);
    if (sinkpad) {
        gst_element_release_request_pad(streammux, sinkpad);
        gst_object_unref(sinkpad);
    }

    gst_bin_remove(GST_BIN(pipeline), streams[stream_id].source);
    streams[stream_id].source = NULL;
    streams[stream_id].active = FALSE;
    active_source_count--;

    g_print("✓ Stream s%d deleted\n", stream_id);

    g_mutex_unlock(&source_lock);
    return FALSE;
}

/* API server thread */
static void*
api_server_thread(void *arg)
{
    int server_fd, client_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};

    /* Create socket */
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        return NULL;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        return NULL;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(API_PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        return NULL;
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        return NULL;
    }

    g_print("*** API Server listening on port %d ***\n", API_PORT);
    g_print("*** Commands: ADD | DEL <stream_id> | STATUS ***\n\n");

    while (1) {
        if ((client_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            continue;
        }

        memset(buffer, 0, sizeof(buffer));
        int valread = read(client_fd, buffer, 1024);

        if (valread > 0) {
            buffer[valread] = '\0';

            /* Remove newline */
            char *newline = strchr(buffer, '\n');
            if (newline) *newline = '\0';

            g_print("API: Received command: %s\n", buffer);

            if (strncmp(buffer, "ADD", 3) == 0) {
                /* Find first free slot */
                gint free_slot = -1;
                g_mutex_lock(&source_lock);
                for (int i = 0; i < NUM_SOURCES; i++) {
                    if (!streams[i].active) {
                        free_slot = i;
                        break;
                    }
                }
                g_mutex_unlock(&source_lock);

                if (free_slot == -1) {
                    const char *response = "ERROR: All stream slots full\n";
                    write(client_fd, response, strlen(response));
                } else {
                    /* Schedule add in main loop */
                    g_idle_add(add_source_idle, GINT_TO_POINTER(free_slot));

                    char response[256];
                    snprintf(response, sizeof(response),
                             "OK: s%d added\nPublish to: rtsp://34.47.221.242:8554/in_s%d\nWatch at: rtsp://localhost:%d/ds-test\n",
                             free_slot, free_slot, streams[free_slot].rtsp_port);
                    write(client_fd, response, strlen(response));
                }
            }
            else if (strncmp(buffer, "DEL ", 4) == 0) {
                int stream_id = atoi(buffer + 4);

                if (stream_id < 0 || stream_id >= NUM_SOURCES) {
                    const char *response = "ERROR: Invalid stream ID\n";
                    write(client_fd, response, strlen(response));
                } else if (!streams[stream_id].active) {
                    const char *response = "ERROR: Stream not active\n";
                    write(client_fd, response, strlen(response));
                } else {
                    /* Schedule delete in main loop */
                    g_idle_add(delete_source_idle, GINT_TO_POINTER(stream_id));

                    char response[128];
                    snprintf(response, sizeof(response), "OK: s%d deleted\n", stream_id);
                    write(client_fd, response, strlen(response));
                }
            }
            else if (strncmp(buffer, "STATUS", 6) == 0) {
                char response[2048];
                int offset = 0;

                g_mutex_lock(&source_lock);
                offset += snprintf(response + offset, sizeof(response) - offset, "Active streams:\n");
                int active_count = 0;
                for (int i = 0; i < NUM_SOURCES; i++) {
                    if (streams[i].active) {
                        offset += snprintf(response + offset, sizeof(response) - offset,
                                         "  s%d: rtsp://localhost:%d/ds-test\n",
                                         i, streams[i].rtsp_port);
                        active_count++;
                    }
                }
                if (active_count == 0) {
                    offset += snprintf(response + offset, sizeof(response) - offset, "  None\n");
                }
                offset += snprintf(response + offset, sizeof(response) - offset,
                                 "Total: %d/%d\n", active_count, NUM_SOURCES);
                g_mutex_unlock(&source_lock);

                write(client_fd, response, strlen(response));
            }
            else {
                const char *response = "ERROR: Unknown command. Use: ADD | DEL <id> | STATUS\n";
                write(client_fd, response, strlen(response));
            }
        }

        close(client_fd);
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    GstElement *pgie, *tracker;
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstRTSPServer *rtsp_server[NUM_SOURCES];
    GstRTSPMountPoints *mounts[NUM_SOURCES];
    GstRTSPMediaFactory *factory[NUM_SOURCES];

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);
    g_mutex_init(&source_lock);

    /* Create pipeline */
    pipeline = gst_pipeline_new("deepstream-multi-pipeline");

    /* Create main elements */
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie = gst_element_factory_make("nvinfer", "primary-inference");
    tracker = gst_element_factory_make("nvtracker", "tracker");
    demux = gst_element_factory_make("nvstreamdemux", "demuxer");

    if (!pipeline || !streammux || !pgie || !tracker || !demux) {
        g_printerr("Failed to create main elements\n");
        return -1;
    }

    /* Configure nvstreammux */
    g_object_set(G_OBJECT(streammux),
                 "width", 720,
                 "height", 1280,
                 "batch-size", NUM_SOURCES,
                 "batched-push-timeout", 33333,
                 "live-source", 1,
                 "sync-inputs", FALSE,
                 NULL);

    g_print("*** DYNAMIC BATCHED BOWLING DETECTION (up to 36 streams) ***\n");
    g_print("*** All streams share 1 YOLO12n model instance ***\n\n");

    /* Configure PGIE */
    g_object_set(G_OBJECT(pgie),
                 "config-file-path", "/config/s0_bowling_batch2.txt",
                 NULL);

    /* Configure Tracker */
    g_object_set(G_OBJECT(tracker),
                 "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                 "ll-config-file", "/config/s0_tracker_iou_optimized.yml",
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

        /* Configure queue */
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

    /* Add main elements to pipeline */
    gst_bin_add_many(GST_BIN(pipeline),
                     streammux, pgie, tracker, demux,
                     NULL);

    /* Add all output chains to pipeline */
    for (int i = 0; i < NUM_SOURCES; i++) {
        gst_bin_add_many(GST_BIN(pipeline),
                         cap_head[i], queue[i], vidconv_pre[i], cap_pre[i], nvosd[i],
                         vidconv_post[i], cap_post[i], encoder[i], parser[i], cap_h264[i],
                         payloader[i], queue_sink[i], sink[i],
                         NULL);
    }

    /* Link main batch inference chain */
    if (!gst_element_link_many(streammux, pgie, tracker, demux, NULL)) {
        g_printerr("Failed to link main batch chain\n");
        return -1;
    }

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
    }

    g_print("✓ RTSP servers ready on ports 8554-8589\n\n");

    /* Set up message bus */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    /* Pre-add all 36 sources */
    g_print("*** Adding all 36 sources (s0-s35) ***\n");

    for (int i = 0; i < NUM_SOURCES; i++) {
        gchar elem_name[32];
        g_snprintf(elem_name, sizeof(elem_name), "source-%d", i);
        streams[i].source = gst_element_factory_make("nvurisrcbin", elem_name);

        if (!streams[i].source) {
            g_printerr("Failed to create source-%d\n", i);
            return -1;
        }

        gchar input_uri[128];
        g_snprintf(input_uri, sizeof(input_uri), "rtsp://34.47.221.242:8554/in_s%d", i);

        g_object_set(G_OBJECT(streams[i].source),
                     "uri", input_uri,
                     "rtsp-reconnect-interval", 10,
                     "rtsp-reconnect-attempts", -1,
                     "latency", 2000,
                     "select-rtp-protocol", 4,
                     NULL);

        g_signal_connect(streams[i].source, "pad-added",
                         G_CALLBACK(on_pad_added), GINT_TO_POINTER(i));

        gst_bin_add(GST_BIN(pipeline), streams[i].source);
        streams[i].active = TRUE;
        active_source_count++;

        if (i % 10 == 0 || i == NUM_SOURCES - 1) {
            g_print("  ✓ Added sources s%d-%d\n", (i/10)*10, i);
        }
    }

    /* Set pipeline to PAUSED then PLAYING */
    g_print("\n*** All 36 sources added - starting pipeline ***\n");
    g_print("*** Setting pipeline to PAUSED ***\n");
    gst_element_set_state(pipeline, GST_STATE_PAUSED);

    g_print("*** Setting pipeline to PLAYING ***\n");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Note: With live sources, transition to PLAYING happens asynchronously */
    /* Pipeline will reach PLAYING once first source starts producing frames */
    g_print("\n*** Pipeline starting with 36 sources ***\n");
    g_print("*** Streams will activate as they connect (s0 already linked) ***\n");
    g_print("*** Watch streams: rtsp://localhost:8554-8589/ds-test ***\n\n");

    /* Run main loop */
    g_main_loop_run(loop);

    /* Clean up */
    g_print("Stopping pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    g_mutex_clear(&source_lock);

    return 0;
}
