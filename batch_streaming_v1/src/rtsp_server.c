// Minimal C RTSP server that builds post-demux branches using NVENC (nvv4l2h264enc)
// and rewraps them to RTSP by UDP (udpsink/udpsrc with H264 RTP). Pre-demux is
// provided as a single-line config string (pipeline.txt) parsed by gst_parse_launch.

#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>

typedef struct {
  // Outputs / serving
  guint streams;       // number of outputs to expose (default 3)
  guint rtsp_port;     // RTSP TCP port (e.g., 8554)
  guint base_udp_port; // base UDP port for per-stream RTP egress (default 5000)

  // Pre-demux (config-driven)
  gchar *pipeline_txt; // path to pre-demux pipeline description (must end with nvstreamdemux name=demux)

  // Post-demux
  gboolean use_osd;    // insert nvosdbin per branch (default on for overlays)

  // Auto-add sample streams via nvmultiurisrcbin REST
  guint auto_add_samples;   // number of sample streams to auto-add (0=disabled)
  guint auto_add_wait_ms;   // initial wait before auto-add posts (ms)
  gchar *sample_uri;        // sample URI to add (defaults to DS sample_1080p)
} AppConfig;

//

static gchar *read_file_to_string(const gchar *path) {
  gchar *contents = NULL; gsize len = 0; GError *err = NULL;
  if (!g_file_get_contents(path, &contents, &len, &err)) {
    g_printerr("Failed to read %s: %s\n", path, err ? err->message : "unknown error");
    if (err) g_error_free(err);
    return NULL;
  }
  g_strstrip(contents);
  return contents;
}

static gboolean parse_args(int argc, char *argv[], AppConfig *cfg) {
  // Defaults
  cfg->streams = 0; // start with no /sN endpoints; add via control API
  cfg->rtsp_port = 8554;
  cfg->pipeline_txt = g_strdup("/opt/nvidia/deepstream/deepstream-8.0/pipeline.txt");
  cfg->use_osd = TRUE;
  cfg->base_udp_port = 5000;
  cfg->auto_add_samples = 0;
  cfg->auto_add_wait_ms = 1000;
  cfg->sample_uri = g_strdup("file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4");

  // CLI (optional; first arg can be a file path for compatibility)
  if (argc >= 2 && g_file_test(argv[1], G_FILE_TEST_EXISTS)) {
    g_free(cfg->pipeline_txt);
    cfg->pipeline_txt = g_strdup(argv[1]);
  }
  if (argc >= 3) cfg->rtsp_port = (guint) g_ascii_strtoull(argv[2], NULL, 10);

  // Environment overrides
  const gchar *env;
  if ((env = g_getenv("RTSP_PORT"))) cfg->rtsp_port = (guint) g_ascii_strtoull(env, NULL, 10);
  if ((env = g_getenv("USE_OSD"))) cfg->use_osd = (gboolean)(g_ascii_strcasecmp(env, "0") != 0);
  if ((env = g_getenv("BASE_UDP_PORT"))) cfg->base_udp_port = (guint) g_ascii_strtoull(env, NULL, 10);
  if ((env = g_getenv("AUTO_ADD_SAMPLES"))) cfg->auto_add_samples = (guint) g_ascii_strtoull(env, NULL, 10);
  if ((env = g_getenv("AUTO_ADD_WAIT_MS"))) cfg->auto_add_wait_ms = (guint) g_ascii_strtoull(env, NULL, 10);
  if ((env = g_getenv("SAMPLE_URI"))) { g_free(cfg->sample_uri); cfg->sample_uri = g_strdup(env); }

  return TRUE;
}

static void cleanup_config(AppConfig *cfg) {
  g_free(cfg->pipeline_txt);
  g_free(cfg->sample_uri);
}

// --- Minimal HTTP POST helper to nvmultiurisrcbin REST (localhost:9010)
static gboolean http_post_localhost_9010(const char *path, const char *json, size_t json_len) {
  struct addrinfo hints; memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo *res = NULL;
  int err = getaddrinfo("127.0.0.1", "9010", &hints, &res);
  if (err != 0 || !res) {
    g_printerr("REST: getaddrinfo failed: %s\n", gai_strerror(err));
    if (res) freeaddrinfo(res);
    return FALSE;
  }
  int s = -1; struct addrinfo *rp;
  for (rp = res; rp != NULL; rp = rp->ai_next) {
    s = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (s == -1) continue;
    if (connect(s, rp->ai_addr, rp->ai_addrlen) == 0) break;
    close(s); s = -1;
  }
  freeaddrinfo(res);
  if (s == -1) { g_printerr("REST: connect failed\n"); return FALSE; }

  gchar *req = g_strdup_printf(
    "POST %s HTTP/1.1\r\n"
    "Host: 127.0.0.1:9010\r\n"
    "Content-Type: application/json\r\n"
    "Content-Length: %zu\r\n"
    "Connection: close\r\n\r\n",
    path, json_len);

  ssize_t n = send(s, req, strlen(req), 0);
  g_free(req);
  if (n < 0) { g_printerr("REST: send header failed\n"); close(s); return FALSE; }
  if (json_len > 0) {
    ssize_t m = send(s, json, json_len, 0);
    if (m < 0) { g_printerr("REST: send body failed\n"); close(s); return FALSE; }
  }
  char buf[512];
  (void)recv(s, buf, sizeof(buf), 0); // best-effort read and ignore
  close(s);
  return TRUE;
}

typedef struct {
  guint count;
  guint delay_ms;
  gchar *sample_uri;
} AutoAddCtx;

static gpointer auto_add_thread(gpointer data) {
  AutoAddCtx *ctx = (AutoAddCtx *)data;
  g_usleep((gulong)ctx->delay_ms * 1000);
  for (guint i = 0; i < ctx->count; ++i) {
    gchar *body = g_strdup_printf(
      "{\n"
      "  \"key\": \"sensor\",\n"
      "  \"value\": {\n"
      "    \"camera_id\": \"auto_%u\",\n"
      "    \"camera_url\": \"%s\",\n"
      "    \"change\": \"camera_add\",\n"
      "    \"metadata\": {\n"
      "      \"resolution\": \"1920x1080\",\n"
      "      \"codec\": \"h264\",\n"
      "      \"framerate\": 30\n"
      "    }\n"
      "  },\n"
      "  \"headers\": { \"source\": \"app\" }\n"
      "}\n",
      i, ctx->sample_uri);
    gboolean ok = http_post_localhost_9010("/api/v1/stream/add", body, strlen(body));
    g_print("REST: add sample %u -> %s\n", i, ok ? "OK" : "FAIL");
    g_free(body);
    g_usleep(300 * 1000); // slight gap between additions
  }
  g_free(ctx->sample_uri);
  g_free(ctx);
  return NULL;
}

static void on_bus_message(GstBus *bus, GstMessage *msg, gpointer user_data) {
  (void)bus; (void)user_data;
  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_ERROR: {
      GError *err = NULL; gchar *dbg = NULL;
      gst_message_parse_error(msg, &err, &dbg);
      g_printerr("ERROR from %s: %s\n", GST_OBJECT_NAME(msg->src), err ? err->message : "(no msg)");
      if (dbg) { g_printerr("Debug: %s\n", dbg); g_free(dbg); }
      if (err) g_error_free(err);
      break;
    }
    case GST_MESSAGE_WARNING: {
      GError *err = NULL; gchar *dbg = NULL;
      gst_message_parse_warning(msg, &err, &dbg);
      g_printerr("WARNING from %s: %s\n", GST_OBJECT_NAME(msg->src), err ? err->message : "(no msg)");
      if (dbg) { g_printerr("Debug: %s\n", dbg); g_free(dbg); }
      if (err) g_error_free(err);
      break;
    }
    case GST_MESSAGE_EOS:
      g_print("Pipeline EOS\n");
      break;
    default:
      break;
  }
}

static gboolean build_full_pipeline_and_server(const AppConfig *cfg, GstPipeline **out_pipeline, GstRTSPServer **out_server) {
  // Obtain pre-demux pipeline description
  gchar *pre_desc = NULL;
  if (cfg->pipeline_txt) pre_desc = read_file_to_string(cfg->pipeline_txt);
  if (!pre_desc) return FALSE;

  GError *err = NULL;
  GstElement *pre = gst_parse_launch(pre_desc, &err);
  g_free(pre_desc);
  if (err) {
    g_printerr("Failed to parse pre-demux pipeline: %s\n", err->message);
    g_error_free(err);
    return FALSE;
  }

  GstPipeline *pipeline = GST_PIPELINE(gst_pipeline_new(NULL));
  gst_bin_add(GST_BIN(pipeline), pre);

  // Find demux
  GstElement *demux = gst_bin_get_by_name(GST_BIN(pipeline), "demux");
  if (!demux) {
    g_printerr("Could not find 'demux' (nvstreamdemux name=demux).\n");
    gst_object_unref(pipeline);
    return FALSE;
  }

  // Create branches per stream: encode + RTP/UDP egress (DeepStream RTSP pattern)
  const guint base_udp = cfg->base_udp_port;
  for (guint i = 0; i < cfg->streams; ++i) {
    GstElement *queue = gst_element_factory_make("queue", NULL);
    GstElement *conv_pre = gst_element_factory_make("nvvideoconvert", NULL);
    GstElement *caps_pre = gst_element_factory_make("capsfilter", NULL);
    GstElement *osd = cfg->use_osd ? gst_element_factory_make("nvosdbin", NULL) : NULL;
    GstElement *conv_post = gst_element_factory_make("nvvideoconvert", NULL);
    GstElement *caps_post = gst_element_factory_make("capsfilter", NULL);
    GstElement *enc = gst_element_factory_make("nvv4l2h264enc", NULL);
    GstElement *parse = gst_element_factory_make("h264parse", NULL);
    GstElement *pay = gst_element_factory_make("rtph264pay", NULL);
    GstElement *udp = gst_element_factory_make("udpsink", NULL);
    if (!queue || !conv_pre || !caps_pre || !conv_post || !caps_post || !enc || !parse || !pay || !udp || (cfg->use_osd && !osd)) {
      g_printerr("Element creation failed for branch %u (pre/osd/post/enc/rtp/udp)\n", i);
      gst_object_unref(demux); gst_object_unref(pipeline);
      return FALSE;
    }
    // Tune queue for low-latency under load
    g_object_set(queue, "leaky", 2, "max-size-time", (guint64)200000000, "max-size-buffers", 0, "max-size-bytes", 0, NULL);

    // Convert to RGBA for OSD, then back to NV12 for encoder
    GstCaps *caps_rgba = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
    g_object_set(caps_pre, "caps", caps_rgba, NULL);
    gst_caps_unref(caps_rgba);
    GstCaps *caps_nv12 = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12,framerate=30/1");
    g_object_set(caps_post, "caps", caps_nv12, NULL);
    gst_caps_unref(caps_nv12);

    // NVENC properties (low-latency friendly)
    g_object_set(enc, "insert-sps-pps", 1, "iframeinterval", 30, "idrinterval", 30, "bitrate", 3000000, NULL);
    // payloader
    g_object_set(pay, "config-interval", 1, "pt", 96, NULL);
    // UDP sink per stream
    guint port = base_udp + i;
    g_object_set(udp, "host", "127.0.0.1", "port", port, "sync", FALSE, "async", FALSE, NULL);

    gst_bin_add_many(GST_BIN(pre), queue, conv_pre, caps_pre, conv_post, caps_post, enc, parse, pay, udp, NULL);
    if (cfg->use_osd) gst_bin_add(GST_BIN(pre), osd);
    gboolean ok = TRUE;
    if (cfg->use_osd) ok = gst_element_link_many(queue, conv_pre, caps_pre, osd, conv_post, caps_post, enc, parse, pay, udp, NULL);
    else ok = gst_element_link_many(queue, conv_pre, caps_pre, conv_post, caps_post, enc, parse, pay, udp, NULL);
    if (!ok) { g_printerr("Link failed for branch %u (pre/osd/post/enc/rtp/udp)\n", i); gst_object_unref(demux); gst_object_unref(pipeline); return FALSE; }

    // Explicitly request and link nvstreamdemux src_%u -> queue
    gchar *padname = g_strdup_printf("src_%u", i);
    GstPad *demux_src = gst_element_request_pad_simple(demux, padname);
    g_free(padname);
    if (!demux_src) {
      g_printerr("Failed to request demux pad for index %u\n", i);
      gst_object_unref(demux); gst_object_unref(pipeline);
      return FALSE;
    }
    GstPad *queue_sink = gst_element_get_static_pad(queue, "sink");
    if (!queue_sink) {
      g_printerr("Queue sink pad missing for index %u\n", i);
      gst_object_unref(demux_src);
      gst_object_unref(demux); gst_object_unref(pipeline);
      return FALSE;
    }
    if (gst_pad_link(demux_src, queue_sink) != GST_PAD_LINK_OK) {
      g_printerr("Failed to link demux src_%u -> queue\n", i);
      gst_object_unref(demux_src); gst_object_unref(queue_sink);
      gst_object_unref(demux); gst_object_unref(pipeline);
      return FALSE;
    }
    gst_object_unref(demux_src);
    gst_object_unref(queue_sink);
    g_print("Linked demux src_%u to UDP egress port %u\n", i, port);
  }

  // RTSP server wrapping UDP ports
  
  GstRTSPServer *server = gst_rtsp_server_new();
  // Listen on all interfaces
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
    g_printerr("Failed to attach RTSP server after retries\n");
    gst_object_unref(demux); gst_object_unref(pipeline);
    return FALSE;
  }

  GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);
  // Optional health/test endpoint that doesn't depend on UDP
  // Use software encoder to avoid GPU/NVMM dependencies here.
  {
    const gchar *launch_test =
      "( videotestsrc is-live=true pattern=smpte "
      "! videoconvert "
      "! jpegenc quality=85 "
      "! rtpjpegpay pt=26 name=pay0 )";
    GstRTSPMediaFactory *f_test = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(f_test, launch_test);
    gst_rtsp_media_factory_set_shared(f_test, TRUE);
    gst_rtsp_media_factory_set_latency(f_test, 100);
    gst_rtsp_mount_points_add_factory(mounts, "/test", f_test);
    g_print("RTSP mounted: rtsp://127.0.0.1:%u/test (synthetic)\n", chosen_port);
  }
  for (guint i = 0; i < cfg->streams; ++i) {
    guint port = 5000 + i;
    gchar *path = g_strdup_printf("/s%u", i);
    gchar *launch = g_strdup_printf(
      "( udpsrc port=%u buffer-size=%lu caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96\" name=pay0 )",
      port, (unsigned long) (524288UL));
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, launch);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_media_factory_set_latency(factory, 100);
    gst_rtsp_mount_points_add_factory(mounts, path, factory);
    g_print("RTSP mounted: rtsp://127.0.0.1:%u%s (udp-wrap H264 RTP @127.0.0.1:%u)\n", chosen_port, path, port);
    g_free(path); g_free(launch);
  }
  g_object_unref(mounts);

  // Demux pads requested and linked above for each branch.

  *out_pipeline = pipeline;
  *out_server = server;
  g_print("Pipeline PLAYING. RTSP server on %u.\n", chosen_port);
  return TRUE;
}

int main(int argc, char *argv[]) {
  AppConfig cfg;
  if (!parse_args(argc, argv, &cfg)) {
    g_printerr("Argument parsing failed\n");
    return 1;
  }

  gst_init(&argc, &argv);

  GstPipeline *pipeline = NULL; GstRTSPServer *server = NULL;
  if (!build_full_pipeline_and_server(&cfg, &pipeline, &server)) {
    cleanup_config(&cfg);
    return 2;
  }

  // Attach bus handler
  GstBus *bus = gst_element_get_bus(GST_ELEMENT(pipeline));
  gst_bus_add_signal_watch(bus);
  g_signal_connect(bus, "message", G_CALLBACK(on_bus_message), NULL);

  // Start pipeline
  gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);
  // Optional: auto-add sample sources via REST
  if (cfg.auto_add_samples > 0) {
    AutoAddCtx *ctx = g_new0(AutoAddCtx, 1);
    ctx->count = cfg.auto_add_samples;
    ctx->delay_ms = cfg.auto_add_wait_ms;
    ctx->sample_uri = g_strdup(cfg.sample_uri);
    GThread *t = g_thread_new("auto_add_samples", auto_add_thread, ctx);
    (void)t;
    g_print("REST: scheduled auto-add of %u sample streams after %u ms\n", cfg.auto_add_samples, cfg.auto_add_wait_ms);
  }
  (void)server; // already logged chosen port during build

  // Main loop
  GMainLoop *loop = g_main_loop_new(NULL, FALSE);
  g_main_loop_run(loop);

  // Teardown
  g_main_loop_unref(loop);
  gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
  if (bus) {
    gst_bus_remove_signal_watch(bus);
    gst_object_unref(bus);
  }

  cleanup_config(&cfg);
  return 0;
}
