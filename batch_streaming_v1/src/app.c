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
GstRTSPServer *g_rtsp_server = NULL;
guint g_rtsp_port = 0;
guint g_next_index = 0;
guint g_base_udp_port_glb = 5000;
GMutex g_state_lock;
guint g_ctrl_port = 0;
const gchar *g_public_host = NULL;
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
  const char *required[] = { "nvstreamdemux", "nvosdbin", "nvvideoconvert", "rtph264pay", "h264parse", "udpsink", NULL };
  for (int i = 0; required[i]; ++i) {
    if (!have_factory(required[i])) {
      LOG_ERR("Missing required GStreamer element: %s", required[i]);
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

// --- Build pipeline + RTSP server (no external pipeline.txt)
gboolean build_full_pipeline_and_server(const AppConfig *cfg, GstPipeline **out_pipeline, GstRTSPServer **out_server) {
  // Build pre-demux chain in one go for readability. It uses:
  // nvmultiurisrcbin ! nvinfer (pgie) ! nvstreamdemux name=demux
  const gchar *pre_desc =
    "nvmultiurisrcbin max-batch-size=64 batched-push-timeout=100000 width=1280 height=720 "
    "file-loop=true sync-inputs=false attach-sys-ts=true drop-on-latency=false "
    "! nvinfer config-file-path=/opt/nvidia/deepstream/deepstream-8.0/pgie.txt "
    "! nvstreamdemux name=demux";

  GError *err = NULL;
  GstElement *pre = gst_parse_launch(pre_desc, &err);
  if (err) {
    LOG_ERR("Failed to parse pre-demux pipeline: %s", err->message);
    g_error_free(err);
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

  // RTSP server wrapping UDP ports
  GstRTSPServer *server = gst_rtsp_server_new();
  gst_rtsp_server_set_address(server, "0.0.0.0");
  guint chosen_port = cfg->rtsp_port;
  guint attach_id = 0;
  for (guint attempt = 0; attempt < 10 && attach_id == 0; ++attempt) {
    gchar *service = g_strdup_printf("%u", chosen_port);
    gst_rtsp_server_set_service(server, service);
    g_free(service);
    attach_id = gst_rtsp_server_attach(server, NULL);
    if (attach_id == 0) chosen_port++;
  }
  if (attach_id == 0) {
    LOG_ERR("Failed to attach RTSP server after retries");
    gst_object_unref(demux); gst_object_unref(pipeline);
    return FALSE;
  }

  GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);
  // Synthetic test endpoint
  {
    const gchar *launch_test =
      "( videotestsrc is-live=true pattern=smpte "
      "! videoconvert ! jpegenc quality=85 ! rtpjpegpay pt=26 name=pay0 )";
    GstRTSPMediaFactory *f_test = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(f_test, launch_test);
    gst_rtsp_media_factory_set_shared(f_test, TRUE);
    gst_rtsp_media_factory_set_latency(f_test, 100);
    gst_rtsp_mount_points_add_factory(mounts, "/test", f_test);
    LOG_INF("RTSP mounted: rtsp://%s:%u/test (synthetic)", g_public_host ? g_public_host : "127.0.0.1", chosen_port);
  }
  g_object_unref(mounts);

  *out_pipeline = pipeline;
  *out_server = server;
  LOG_INF("Pipeline READY. RTSP server on %u", chosen_port);
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

  GstPipeline *pipeline = NULL; GstRTSPServer *server = NULL;
  if (!build_full_pipeline_and_server(cfg, &pipeline, &server)) return FALSE;

  // Attach bus handler for logs
  GstBus *bus = gst_element_get_bus(GST_ELEMENT(pipeline));
  gst_bus_add_signal_watch(bus);
  g_signal_connect(bus, "message", G_CALLBACK(on_bus_message), NULL);

  // Start pipeline
  gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);

  // Stash globals for dynamic add
  g_mutex_init(&g_state_lock);
  g_pipeline = pipeline;
  g_rtsp_server = server;
  g_rtsp_port = cfg->rtsp_port; // will be updated below from server service
  g_base_udp_port_glb = cfg->base_udp_port;
  g_demux = gst_bin_get_by_name(GST_BIN(pipeline), "demux");
  g_pre_bin = GST_ELEMENT(gst_element_get_parent(g_demux));
  g_next_index = 0;
  g_public_host = cfg->public_host;
  const gchar *service = gst_rtsp_server_get_service(server);
  if (service) g_rtsp_port = (guint) g_ascii_strtoull(service, NULL, 10);

  // Start minimal control API
  (void)g_thread_new("ctrl_http", control_http_thread, NULL);

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
