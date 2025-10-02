// Per-stream branch building (L2)
#include "log.h"
#include "state.h"
#include <gst/gst.h>
#include <string.h>

typedef struct {
  GstElement *queue, *conv_pre, *caps_pre, *osd, *conv_post, *caps_post;
  // SW path throttle + format
  GstElement *conv_cpu, *rate_sw, *caps_cpu;
  // HW path throttle sandwich (sysmem fps then back to NVMM)
  GstElement *conv_rate_hw, *caps_rate_sys_hw, *rate_hw, *caps_rate_fps_hw, *conv_back_hw, *caps_back_nvmm_hw;
  // Common tail
  GstElement *enc, *parse, *pay, *udp;
} BranchElems;

static guint get_env_uint(const char *name, guint defval) {
  const gchar *e = g_getenv(name);
  if (!e || !*e) return defval;
  return (guint) g_ascii_strtoull(e, NULL, 10);
}

static gboolean create_elements(guint index, BranchElems *e, gboolean *enc_is_hw, gboolean *enc_is_x264, guint *out_port) {
  memset(e, 0, sizeof(*e));
  *enc_is_hw = FALSE;
  *enc_is_x264 = FALSE;

  e->queue = gst_element_factory_make("queue", NULL);
  e->conv_pre = gst_element_factory_make("nvvideoconvert", NULL);
  e->caps_pre = gst_element_factory_make("capsfilter", NULL);
  e->osd = gst_element_factory_make("nvosdbin", NULL);
  e->conv_post = gst_element_factory_make("nvvideoconvert", NULL);
  e->caps_post = gst_element_factory_make("capsfilter", NULL);
  e->conv_cpu = gst_element_factory_make("nvvideoconvert", NULL);
  e->rate_sw = gst_element_factory_make("videorate", NULL);
  e->caps_cpu = gst_element_factory_make("capsfilter", NULL);
  // HW throttle chain
  e->conv_rate_hw = gst_element_factory_make("nvvideoconvert", NULL);
  e->caps_rate_sys_hw = gst_element_factory_make("capsfilter", NULL);
  e->rate_hw = gst_element_factory_make("videorate", NULL);
  e->caps_rate_fps_hw = gst_element_factory_make("capsfilter", NULL);
  e->conv_back_hw = gst_element_factory_make("nvvideoconvert", NULL);
  e->caps_back_nvmm_hw = gst_element_factory_make("capsfilter", NULL);

  // Encoder selection (HW for first g_hw_threshold)
  if (index < g_hw_threshold) {
    e->enc = gst_element_factory_make("nvv4l2h264enc", NULL);
    *enc_is_hw = (e->enc != NULL);
  }
  if (!e->enc) {
    e->enc = gst_element_factory_make("x264enc", NULL);
    *enc_is_x264 = (e->enc != NULL);
    if (*enc_is_x264) LOG_WRN("Using software encoder: x264enc (index=%u)", index);
  }
  if (!e->enc) {
    e->enc = gst_element_factory_make("avenc_h264", NULL);
    if (e->enc) LOG_WRN("Using software encoder fallback: avenc_h264 (index=%u)", index);
  }
  if (!e->enc) {
    e->enc = gst_element_factory_make("openh264enc", NULL);
    if (e->enc) LOG_WRN("Using software encoder fallback: openh264enc (index=%u)", index);
  }

  e->parse = gst_element_factory_make("h264parse", NULL);
  e->pay = gst_element_factory_make("rtph264pay", NULL);
  e->udp = gst_element_factory_make("udpsink", NULL);

  if (!e->queue || !e->conv_pre || !e->caps_pre || !e->osd || !e->conv_post || !e->caps_post || !e->enc || !e->parse || !e->pay || !e->udp) {
    LOG_ERR("Element creation failed for /s%u", index);
    return FALSE;
  }

  // Properties and caps
  g_object_set(e->queue, "leaky", 0, "max-size-time", (guint64)200000000, "max-size-buffers", 0, "max-size-bytes", 0, NULL);
  GstCaps *caps_rgba = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
  g_object_set(e->caps_pre, "caps", caps_rgba, NULL);
  gst_caps_unref(caps_rgba);
  GstCaps *caps_nv12 = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
  g_object_set(e->caps_post, "caps", caps_nv12, NULL);
  gst_caps_unref(caps_nv12);

  if (*enc_is_hw) {
    g_object_set(e->enc,
      "insert-sps-pps", 1,
      "iframeinterval", 30,
      "idrinterval", 30,
      "bitrate", 3000000,
      "maxperf-enable", 1,
      "control-rate", 1,
      "preset-level", 1,
      NULL);
    // Prepare HW fps throttle elements (sysmem NV12 -> videorate -> NVMM NV12)
    guint fps = get_env_uint("OUTPUT_FPS", 30);
    if (!e->conv_rate_hw || !e->caps_rate_sys_hw || !e->rate_hw || !e->caps_rate_fps_hw || !e->conv_back_hw || !e->caps_back_nvmm_hw) {
      LOG_ERR("Element creation failed (HW throttle) for /s%u", index);
      return FALSE;
    }
    GstCaps *caps_sys = gst_caps_from_string("video/x-raw,format=NV12");
    g_object_set(e->caps_rate_sys_hw, "caps", caps_sys, NULL);
    gst_caps_unref(caps_sys);
    gchar *fps_caps = g_strdup_printf("video/x-raw,framerate=%u/1", fps);
    GstCaps *caps_fps = gst_caps_from_string(fps_caps);
    g_free(fps_caps);
    g_object_set(e->caps_rate_fps_hw, "caps", caps_fps, NULL);
    gst_caps_unref(caps_fps);
    GstCaps *caps_nvmm_back = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
    g_object_set(e->caps_back_nvmm_hw, "caps", caps_nvmm_back, NULL);
    gst_caps_unref(caps_nvmm_back);
  } else {
    guint fps = get_env_uint("OUTPUT_FPS", 30);
    if (!e->rate_sw) {
      LOG_ERR("Missing 'videorate' for SW path /s%u", index);
      return FALSE;
    }
    gchar *caps_str = g_strdup_printf("video/x-raw,format=I420,framerate=%u/1", fps);
    GstCaps *caps_i420 = gst_caps_from_string(caps_str);
    g_free(caps_str);
    g_object_set(e->caps_cpu, "caps", caps_i420, NULL);
    gst_caps_unref(caps_i420);
    if (*enc_is_x264) {
      guint cores = (guint) g_get_num_processors();
      guint def_threads = cores > 1 ? (cores / 2) : 1; if (def_threads > 4) def_threads = 4; if (def_threads < 1) def_threads = 1;
      guint threads = get_env_uint("X264_THREADS", def_threads);
      g_object_set(e->enc, "tune", "zerolatency", "speed-preset", "ultrafast", "bitrate", 3000, "key-int-max", 60, "bframes", 0, "threads", threads, NULL);
      GObjectClass *klass = G_OBJECT_GET_CLASS(e->enc);
      if (g_object_class_find_property(klass, "sliced-threads")) {
        g_object_set(e->enc, "sliced-threads", TRUE, NULL);
      }
    } else {
      g_object_set(e->enc, "bitrate", 3000000, NULL);
    }
  }
  g_object_set(e->pay, "config-interval", 1, "pt", 96, NULL);
  guint port = g_base_udp_port_glb + index;
  g_object_set(e->udp, "host", "127.0.0.1", "port", port, "sync", FALSE, "async", FALSE, NULL);
  if (out_port) *out_port = port;
  return TRUE;
}

static void cleanup_branch(const BranchElems *e, gboolean enc_is_hw) {
  if (!e) return;
  if (enc_is_hw) {
    gst_bin_remove_many(GST_BIN(g_pre_bin), e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post,
      e->conv_rate_hw, e->caps_rate_sys_hw, e->rate_hw, e->caps_rate_fps_hw, e->conv_back_hw, e->caps_back_nvmm_hw,
      e->enc, e->parse, e->pay, e->udp, NULL);
  } else {
    gst_bin_remove_many(GST_BIN(g_pre_bin), e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->conv_cpu, e->rate_sw, e->caps_cpu, e->enc, e->parse, e->pay, e->udp, NULL);
  }
}

static gboolean link_branch(guint index, const BranchElems *e, gboolean enc_is_hw) {
  gboolean ok = FALSE;
  if (enc_is_hw) {
    gst_bin_add_many(GST_BIN(g_pre_bin), e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post,
      e->conv_rate_hw, e->caps_rate_sys_hw, e->rate_hw, e->caps_rate_fps_hw, e->conv_back_hw, e->caps_back_nvmm_hw,
      e->enc, e->parse, e->pay, e->udp, NULL);
    ok = gst_element_link_many(e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post,
      e->conv_rate_hw, e->caps_rate_sys_hw, e->rate_hw, e->caps_rate_fps_hw, e->conv_back_hw, e->caps_back_nvmm_hw,
      e->enc, e->parse, e->pay, e->udp, NULL);
  } else {
    gst_bin_add_many(GST_BIN(g_pre_bin), e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->conv_cpu, e->rate_sw, e->caps_cpu, e->enc, e->parse, e->pay, e->udp, NULL);
    ok = gst_element_link_many(e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->conv_cpu, e->rate_sw, e->caps_cpu, e->enc, e->parse, e->pay, e->udp, NULL);
  }
  if (!ok) return FALSE;

  gchar *padname = g_strdup_printf("src_%u", index);
  GstPad *demux_src = gst_element_request_pad_simple(g_demux, padname);
  g_free(padname);
  if (!demux_src) return FALSE;
  GstPad *queue_sink = gst_element_get_static_pad(e->queue, "sink");
  if (!queue_sink) { gst_object_unref(demux_src); return FALSE; }
  if (gst_pad_link(demux_src, queue_sink) != GST_PAD_LINK_OK) {
    gst_object_unref(demux_src); gst_object_unref(queue_sink); return FALSE;
  }
  gst_object_unref(demux_src); gst_object_unref(queue_sink);

  if (enc_is_hw) {
    GstElement *els[] = { e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post,
      e->conv_rate_hw, e->caps_rate_sys_hw, e->rate_hw, e->caps_rate_fps_hw, e->conv_back_hw, e->caps_back_nvmm_hw,
      e->enc, e->parse, e->pay, e->udp, NULL };
    for (int i = 0; els[i]; ++i) gst_element_sync_state_with_parent(els[i]);
  } else {
    GstElement *els[] = { e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->conv_cpu, e->rate_sw, e->caps_cpu, e->enc, e->parse, e->pay, e->udp, NULL };
    for (int i = 0; els[i]; ++i) gst_element_sync_state_with_parent(els[i]);
  }
  return TRUE;
}

static gboolean mount_rtsp(const gchar *path, guint port) {
  // Wrap the already-RTP H264 UDP stream by depayloading and re-payloading so the RTSP
  // factory exposes a proper payloader named pay0 (as expected by gst-rtsp-server).
  gchar *launch = g_strdup_printf(
    "( udpsrc port=%u buffer-size=%lu caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96\" "
    "! rtph264depay ! rtph264pay name=pay0 pt=96 )",
    port, (unsigned long)(524288UL));
  GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(g_rtsp_server);
  GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();
  gst_rtsp_media_factory_set_launch(factory, launch);
  gst_rtsp_media_factory_set_shared(factory, TRUE);
  gst_rtsp_media_factory_set_latency(factory, 100);
  gst_rtsp_mount_points_add_factory(mounts, path, factory);
  g_object_unref(mounts);
  g_free(launch);
  return TRUE;
}

gboolean add_branch_and_mount(guint index, gchar **out_path, gchar **out_url) {
  g_mutex_lock(&g_state_lock);
  if (!g_pipeline || !g_demux || !g_pre_bin || !g_rtsp_server) {
    g_mutex_unlock(&g_state_lock);
    return FALSE;
  }

  BranchElems be; gboolean enc_is_hw = FALSE; gboolean enc_is_x264 = FALSE; guint port = 0;
  if (!create_elements(index, &be, &enc_is_hw, &enc_is_x264, &port)) { g_mutex_unlock(&g_state_lock); return FALSE; }
  if (!link_branch(index, &be, enc_is_hw)) {
    LOG_ERR("Link failed for /s%u (pre/osd/post/cpu/enc/rtp/udp)", index);
    cleanup_branch(&be, enc_is_hw);
    g_mutex_unlock(&g_state_lock);
    return FALSE;
  }

  LOG_INF("Linked demux src_%u to UDP egress port %u (dynamic)", index, port);

  gchar *path = g_strdup_printf("/s%u", index);
  (void)mount_rtsp(path, port);
  LOG_INF("RTSP mounted: rtsp://%s:%u%s (udp-wrap H264 RTP @127.0.0.1:%u)", g_public_host ? g_public_host : "127.0.0.1", g_rtsp_port, path, port);
  if (out_path) *out_path = g_strdup(path);
  if (out_url) *out_url = g_strdup_printf("rtsp://%s:%u%s", g_public_host ? g_public_host : "127.0.0.1", g_rtsp_port, path);

  g_streams[index].in_use = TRUE;
  g_streams[index].enc_is_hw = enc_is_hw;
  strncpy(g_streams[index].enc_kind, enc_is_hw ? "nvenc" : (enc_is_x264 ? "x264" : (g_str_has_prefix(G_OBJECT_TYPE_NAME(be.enc), "GstAv") ? "avenc" : "openh264")), sizeof(g_streams[index].enc_kind)-1);
  g_streams[index].enc_kind[sizeof(g_streams[index].enc_kind)-1] = '\0';
  g_streams[index].udp_port = port;
  g_snprintf(g_streams[index].path, sizeof(g_streams[index].path), "%s", path);
  g_streams[index].queue = be.queue;
  g_streams[index].conv_pre = be.conv_pre;
  g_streams[index].caps_pre = be.caps_pre;
  g_streams[index].osd = be.osd;
  g_streams[index].conv_post = be.conv_post;
  g_streams[index].caps_post = be.caps_post;
  g_streams[index].conv_cpu = enc_is_hw ? NULL : be.conv_cpu;
  g_streams[index].caps_cpu = enc_is_hw ? NULL : be.caps_cpu;
  g_streams[index].enc = be.enc;
  g_streams[index].parse = be.parse;
  g_streams[index].pay = be.pay;
  g_streams[index].udp = be.udp;

  g_free(path);
  g_mutex_unlock(&g_state_lock);
  return TRUE;
}
