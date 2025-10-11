// App lifecycle and RTSP/pipeline setup (L1)
#include <glib-unix.h>
#include <signal.h>
#include "log.h"
#include "state.h"
#include "config.h"
#include "branch.h"
#include "control.h"

// Define shared state (declared in state.h)
GstPipeline *g_pipeline = NULL;
GstElement  *g_pre_bin = NULL;
GstElement  *g_demux = NULL;
guint g_next_index = 0;
GMutex g_state_lock;
guint g_ctrl_port = 0;
const gchar *g_push_url_tmpl = NULL;
guint g_hw_threshold = 8;
guint g_sw_max = 56;
guint g_max_streams = 64;
StreamInfo g_streams[64];

static GMainLoop *g_loop = NULL;

// --- Sanity checks
static gboolean have_factory(const char *name) {
  GstElementFactory *f = gst_element_factory_find(name);
  if (f) { gst_object_unref(f); return TRUE; }
  return FALSE;
}

gboolean sanity_check_plugins(void) {
  const char *required[] = { "nvstreamdemux", "nvosdbin", "nvvideoconvert", "rtph264pay", "h264parse", "rtspclientsink", NULL };
  for (int i = 0; required[i]; ++i) {
    if (!have_factory(required[i])) {
      LOG_ERR("Missing required GStreamer element: %s", required[i]);
      return FALSE;
    }
  }
  // If direct RTSP push is enabled via env, require rtspclientsink
  const gchar *push_tmpl = g_getenv("RTSP_PUSH_URL_TMPL");
  if (push_tmpl && *push_tmpl) {
    if (!have_factory("rtspclientsink")) {
      LOG_ERR("Direct RTSP push requested but rtspclientsink is missing");
      return FALSE;
    }
  }
  if (!have_factory("nvv4l2h264enc")) {
    LOG_WRN("NVENC encoder not found; hardware streams will not be available");
  }
  if (!(have_factory("x264enc") || have_factory("avenc_h264") || have_factory("openh264enc"))) {
    LOG_ERR("No software H.264 encoder found (x264enc|avenc_h264|openh264enc)");
    return FALSE;
  }
  return TRUE;
}

// --- Stream EOS handler for RTSP reconnection
static void handle_stream_eos(guint stream_id) {
  g_mutex_lock(&g_state_lock);

  if (stream_id >= 64 || !g_streams[stream_id].in_use) {
    g_mutex_unlock(&g_state_lock);
    return;
  }

  g_streams[stream_id].eos = TRUE;
  g_streams[stream_id].reconnect_count++;

  LOG_WRN("Stream %u EOS detected (reconnect attempt #%u) - triggering manual reconnection",
          stream_id, g_streams[stream_id].reconnect_count);

  // Send flush-stop to demux sink pad to clear EOS state
  gchar *pad_name = g_strdup_printf("sink_%u", stream_id);
  GstPad *sinkpad = gst_element_get_static_pad(g_demux, pad_name);
  if (sinkpad) {
    gboolean sent = gst_pad_send_event(sinkpad, gst_event_new_flush_stop(FALSE));
    LOG_INF("Sent flush-stop to %s: %s", pad_name, sent ? "OK" : "FAILED");
    gst_object_unref(sinkpad);
  } else {
    LOG_WRN("Could not find pad %s for flush-stop", pad_name);
  }
  g_free(pad_name);

  // Warn if stream has failed too many times
  if (g_streams[stream_id].reconnect_count > 10) {
    LOG_ERR("Stream %u has failed %u times - may need manual intervention",
            stream_id, g_streams[stream_id].reconnect_count);
  }

  // Manual reconnection: add new stream with unique camera_id
  // Each reconnection gets a new source-id, but we keep outputting to same path
  gchar *camera_id = g_strdup_printf("s%u_r%u_%ld", stream_id, g_streams[stream_id].reconnect_count, time(NULL));
  gchar *input_uri = g_strdup_printf("rtsp://34.100.230.7:8554/in_s%u", stream_id);

  LOG_INF("Reconnecting stream %u with camera_id=%s", stream_id, camera_id);

  gchar *add_json = g_strdup_printf(
    "{\"value\":{\"camera_id\":\"%s\",\"camera_url\":\"%s\",\"change\":\"camera_add\"}}",
    camera_id, input_uri);
  gboolean added = nvmulti_rest_post("/api/v1/stream/add", add_json, strlen(add_json));
  g_free(add_json);

  if (added) {
    LOG_INF("Reconnection stream added with camera_id=%s", camera_id);
    g_streams[stream_id].eos = FALSE;  // Clear EOS flag
  } else {
    LOG_ERR("Failed to add reconnection stream for %u", stream_id);
  }

  g_free(camera_id);
  g_free(input_uri);

  g_mutex_unlock(&g_state_lock);
}

// --- Bus logging
static void on_bus_message(GstBus *bus, GstMessage *msg, gpointer user_data) {
  (void)bus; (void)user_data;
  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_ERROR: {
      GError *err = NULL; gchar *dbg = NULL;
      gst_message_parse_error(msg, &err, &dbg);
      LOG_ERR("from %s: %s", GST_OBJECT_NAME(msg->src), err ? err->message : "(no msg)");
      if (dbg) { LOG_WRN("Debug: %s", dbg); g_free(dbg); }
      if (err) g_error_free(err);
      break;
    }
    case GST_MESSAGE_WARNING: {
      GError *err = NULL; gchar *dbg = NULL;
      gst_message_parse_warning(msg, &err, &dbg);
      LOG_WRN("from %s: %s", GST_OBJECT_NAME(msg->src), err ? err->message : "(no msg)");
      if (dbg) { LOG_WRN("Debug: %s", dbg); g_free(dbg); }
      if (err) g_error_free(err);
      break;
    }
    case GST_MESSAGE_EOS:
      LOG_INF("Pipeline EOS");
      break;
    case GST_MESSAGE_ELEMENT: {
      const GstStructure *s = gst_message_get_structure(msg);
      if (s) {
        gchar *struct_str = gst_structure_to_string(s);
        LOG_INF("ELEMENT message from %s: %s", GST_OBJECT_NAME(msg->src), struct_str);
        g_free(struct_str);

        if (gst_structure_has_name(s, "stream-add")) {
          guint source_id;
          if (gst_structure_get_uint(s, "source-id", &source_id)) {
            LOG_INF("Auto-creating output branch for source-id %u", source_id);
            gchar *path = NULL, *url = NULL;
            if (add_branch_and_mount(source_id, &path, &url)) {
              LOG_INF("Created branch: %s -> %s", path, url);
              g_free(path); g_free(url);
            } else {
              LOG_ERR("Failed to create branch for source-id %u", source_id);
            }
          }
        }

        if (gst_structure_has_name(s, "GstRTSPSrcTimeout")) {
          LOG_WRN("RTSP timeout detected - triggering manual reconnection");
          // For now, reconnect camera_id 0 (TODO: map source to camera_id)
          handle_stream_eos(0);
        }

        if (gst_structure_has_name(s, "stream-eos")) {
          guint stream_id;
          if (gst_structure_get_uint(s, "stream-id", &stream_id)) {
            handle_stream_eos(stream_id);
          }
        }
      }
      break;
    }
    default:
      break;
  }
}

// --- Capacity (hard limits)
void decide_max_streams(void) {
  g_hw_threshold = 8;
  g_sw_max = 56;
  g_max_streams = g_hw_threshold + g_sw_max;

  const gchar *env_hw = g_getenv("HW_THRESHOLD");
  const gchar *env_sw = g_getenv("SW_MAX");
  const gchar *env_total = g_getenv("MAX_STREAMS");
  if (env_hw) g_hw_threshold = (guint) g_ascii_strtoull(env_hw, NULL, 10);
  if (env_sw) g_sw_max = (guint) g_ascii_strtoull(env_sw, NULL, 10);
  g_max_streams = g_hw_threshold + g_sw_max;
  if (env_total) g_max_streams = (guint) g_ascii_strtoull(env_total, NULL, 10);

  LOG_INF("Capacity (hard limits): HW=%u, SW=%u, total=%u", g_hw_threshold, g_sw_max, g_max_streams);
}

// --- Build pipeline (no external pipeline.txt)
// Optional: read pipeline description from a file (pipeline.txt) if present.
// If not found, fall back to a sensible default.
static gchar* read_file_to_string(const gchar *path) {
  gchar *contents = NULL; gsize len = 0; GError *e = NULL;
  if (!g_file_get_contents(path, &contents, &len, &e)) {
    if (e) g_error_free(e);
    return NULL;
  }
  g_strstrip(contents);
  return contents;
}

static gchar* get_pipeline_file_path(void) {
  const gchar *env = g_getenv("PIPELINE_FILE");
  if (env && *env) return g_strdup(env);
  return g_strdup("/opt/nvidia/deepstream/deepstream-8.0/pipeline.txt");
}

static guint count_uri_list_items(const gchar *desc) {
  if (!desc) return 0;
  const gchar *p = g_strstr_len(desc, -1, "uri-list=");
  if (!p) return 0;
  p += strlen("uri-list=");
  if (*p != '\'' && *p != '"') return 0;
  gchar quote = *p++;
  const gchar *q = strchr(p, quote);
  if (!q || q <= p) return 0;
  gchar *uris = g_strndup(p, (gsize)(q - p));
  guint count = 0;
  gchar **tokens = g_strsplit(uris, ",", -1);
  for (guint i = 0; tokens && tokens[i]; ++i) {
    if (tokens[i][0] != '\0') count++;
  }
  g_strfreev(tokens);
  g_free(uris);
  return count;
}

gboolean build_pipeline(const AppConfig *cfg, GstPipeline **out_pipeline) {
  (void)cfg; // config not needed to build pre-demux; push is handled per-branch
  // Build pre-demux chain, preferring a file-based config when available.
  // Expected shape: nvmultiurisrcbin [uri-list=...] ! nvinfer ... ! nvstreamdemux name=demux
  gchar *pre_desc_from_file = NULL;
  gchar *pipeline_path = get_pipeline_file_path();
  pre_desc_from_file = read_file_to_string(pipeline_path);
  if (pre_desc_from_file) {
    LOG_INF("Using pipeline file: %s", pipeline_path);
  }
  g_free(pipeline_path);

  const gchar *pre_desc = pre_desc_from_file ? pre_desc_from_file :
    "nvmultiurisrcbin max-batch-size=64 batched-push-timeout=33000 width=1280 height=720 "
    "live-source=1 file-loop=true sync-inputs=false attach-sys-ts=true drop-on-latency=false "
    "! nvinfer config-file-path=/opt/nvidia/deepstream/deepstream-8.0/pgie.txt "
    "! nvstreamdemux name=demux";

  GError *err = NULL;
  GstElement *pre = gst_parse_launch(pre_desc, &err);
  if (err) {
    LOG_ERR("Failed to parse pre-demux pipeline: %s", err->message);
    g_error_free(err);
    g_free(pre_desc_from_file);
    return FALSE;
  }

  GstPipeline *pipeline = GST_PIPELINE(gst_pipeline_new(NULL));
  gst_bin_add(GST_BIN(pipeline), pre);

  // Find demux
  GstElement *demux = gst_bin_get_by_name(GST_BIN(pipeline), "demux");
  if (!demux) {
    LOG_ERR("Could not find 'demux' (nvstreamdemux name=demux)");
    gst_object_unref(pipeline);
    return FALSE;
  }

  g_free(pre_desc_from_file);
  *out_pipeline = pipeline;
  LOG_INF("Pipeline READY. Direct RTSP push mode (no local RTSP server)");
  return TRUE;
}

// --- Signals
static gboolean on_signal_cb(gpointer data) {
  GMainLoop *loop = (GMainLoop*)data;
  LOG_INF("Signal received; shutting down");
  g_main_loop_quit(loop);
  return G_SOURCE_CONTINUE;
}

// --- App lifecycle
gboolean app_setup(const AppConfig *cfg) {
  decide_max_streams();

  GstPipeline *pipeline = NULL;
  if (!build_pipeline(cfg, &pipeline)) return FALSE;

  // Attach bus handler for logs
  GstBus *bus = gst_element_get_bus(GST_ELEMENT(pipeline));
  gst_bus_add_signal_watch(bus);
  g_signal_connect(bus, "message", G_CALLBACK(on_bus_message), NULL);

  // Stash globals for dynamic add
  g_mutex_init(&g_state_lock);
  g_pipeline = pipeline;
  g_demux = gst_bin_get_by_name(GST_BIN(pipeline), "demux");
  g_pre_bin = GST_ELEMENT(gst_element_get_parent(g_demux));
  g_next_index = 0;
  g_push_url_tmpl = cfg->push_url_tmpl; // enable direct RTSP push when set

  // No static bootstrap; start with 0 streams; add streams dynamically via nvmultiurisrcbin REST API
  LOG_INF("Starting with 0 streams; add streams via nvmultiurisrcbin REST API on port 9000");

  // Start pipeline after branches are prepared to reduce early data flow warnings
  gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);

  // Create main loop + signals
  g_loop = g_main_loop_new(NULL, FALSE);
  g_unix_signal_add(SIGINT, on_signal_cb, g_loop);
  g_unix_signal_add(SIGTERM, on_signal_cb, g_loop);
  return TRUE;
}

void app_loop(void) {
  if (g_loop) g_main_loop_run(g_loop);
}

void app_teardown(void) {
  if (g_loop) { g_main_loop_unref(g_loop); g_loop = NULL; }
  if (g_pipeline) {
    gst_element_set_state(GST_ELEMENT(g_pipeline), GST_STATE_NULL);
  }
}
