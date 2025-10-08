// Per-stream branch building (L2)
#include "log.h"
#include "state.h"
#include <gst/gst.h>
#include <string.h>

typedef struct {
  GstElement *queue, *conv_pre, *caps_pre, *osd, *conv_post, *caps_post;
  GstElement *conv_cpu, *caps_cpu, *enc, *parse;
  GstElement *rtsp; // direct RTSP push (rtspclientsink)
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
  e->caps_cpu = gst_element_factory_make("capsfilter", NULL);

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
  e->rtsp = gst_element_factory_make("rtspclientsink", NULL);

  if (!e->queue || !e->conv_pre || !e->caps_pre || !e->osd || !e->conv_post || !e->caps_post || !e->enc || !e->parse || !e->rtsp) {
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
    // Set only properties that exist on this platform's nvv4l2h264enc
    GObjectClass *klass = G_OBJECT_GET_CLASS(e->enc);
    if (g_object_class_find_property(klass, "insert-sps-pps"))
      g_object_set(e->enc, "insert-sps-pps", 1, NULL);
    if (g_object_class_find_property(klass, "iframeinterval"))
      g_object_set(e->enc, "iframeinterval", 30, NULL);
    if (g_object_class_find_property(klass, "idrinterval"))
      g_object_set(e->enc, "idrinterval", 30, NULL);
    if (g_object_class_find_property(klass, "bitrate"))
      g_object_set(e->enc, "bitrate", 3000000, NULL);
    if (g_object_class_find_property(klass, "control-rate"))
      g_object_set(e->enc, "control-rate", 1, NULL);
    if (g_object_class_find_property(klass, "preset-level"))
      g_object_set(e->enc, "preset-level", 1, NULL);
    // Some versions expose 'preset' instead of 'preset-level'
    if (g_object_class_find_property(klass, "preset"))
      g_object_set(e->enc, "preset", 1, NULL);
  } else {
    GstCaps *caps_i420 = gst_caps_from_string("video/x-raw,format=I420");
    g_object_set(e->caps_cpu, "caps", caps_i420, NULL);
    gst_caps_unref(caps_i420);
    if (*enc_is_x264) {
      guint cores = (guint) g_get_num_processors();
      guint def_threads = cores > 1 ? (cores / 2) : 1; if (def_threads > 4) def_threads = 4; if (def_threads < 1) def_threads = 1;
      guint threads = get_env_uint("X264_THREADS", def_threads);
      g_object_set(e->enc, "tune", "zerolatency", "speed-preset", "ultrafast", "bitrate", 3000, "key-int-max", 30, "bframes", 0, "threads", threads, NULL);
      GObjectClass *klass = G_OBJECT_GET_CLASS(e->enc);
      if (g_object_class_find_property(klass, "sliced-threads")) {
        g_object_set(e->enc, "sliced-threads", TRUE, NULL);
      }
    } else {
      g_object_set(e->enc, "bitrate", 3000000, NULL);
    }
  }
  // Prepare H.264 bitstream; rtspclientsink will choose and configure the payloader internally.
  // Direct RTSP push target (location derived from template). If the template has no
  // printf-style specifiers, duplicate it verbatim.
  const gchar *tmpl = (g_push_url_tmpl && *g_push_url_tmpl) ? g_push_url_tmpl : "rtsp://127.0.0.1:8554/s%u";
  gchar *loc = NULL;
  if (strchr(tmpl, '%')) loc = g_strdup_printf(tmpl, index);
  else loc = g_strdup(tmpl);
  // Force TCP for RTSP transport to match prior ffmpeg behavior and avoid UDP traversal issues.
  // protocols flag: GST_RTSP_LOWER_TRANS_TCP = 4
  g_object_set(e->rtsp, "location", loc, "protocols", 4, NULL);
  g_free(loc);
  if (out_port) *out_port = 0;
  return TRUE;
}

static void cleanup_branch(const BranchElems *e, gboolean enc_is_hw) {
  if (!e) return;
  if (enc_is_hw) {
    gst_bin_remove_many(GST_BIN(g_pre_bin), e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->enc, e->parse, e->rtsp, NULL);
  } else {
    gst_bin_remove_many(GST_BIN(g_pre_bin), e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->conv_cpu, e->caps_cpu, e->enc, e->parse, e->rtsp, NULL);
  }
}

static gboolean link_branch(guint index, const BranchElems *e, gboolean enc_is_hw) {
  gboolean ok = FALSE;
  if (enc_is_hw) {
    gst_bin_add_many(GST_BIN(g_pre_bin), e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->enc, e->parse, e->rtsp, NULL);
    ok = gst_element_link_many(e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->enc, e->parse, e->rtsp, NULL);
  } else {
    gst_bin_add_many(GST_BIN(g_pre_bin), e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->conv_cpu, e->caps_cpu, e->enc, e->parse, e->rtsp, NULL);
    ok = gst_element_link_many(e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->conv_cpu, e->caps_cpu, e->enc, e->parse, e->rtsp, NULL);
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
    GstElement *els[] = { e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->enc, e->parse, e->rtsp, NULL };
    for (int i = 0; els[i]; ++i) gst_element_sync_state_with_parent(els[i]);
  } else {
    GstElement *els[] = { e->queue, e->conv_pre, e->caps_pre, e->osd, e->conv_post, e->caps_post, e->conv_cpu, e->caps_cpu, e->enc, e->parse, e->rtsp, NULL };
    for (int i = 0; els[i]; ++i) gst_element_sync_state_with_parent(els[i]);
  }
  return TRUE;
}

gboolean add_branch_and_mount(guint index, gchar **out_path, gchar **out_url) {
  g_mutex_lock(&g_state_lock);
  if (!g_pipeline || !g_demux || !g_pre_bin || !(g_push_url_tmpl && *g_push_url_tmpl)) {
    g_mutex_unlock(&g_state_lock);
    return FALSE;
  }

  BranchElems be; gboolean enc_is_hw = FALSE; gboolean enc_is_x264 = FALSE; guint port = 0;
  if (!create_elements(index, &be, &enc_is_hw, &enc_is_x264, &port)) { g_mutex_unlock(&g_state_lock); return FALSE; }
  if (!link_branch(index, &be, enc_is_hw)) {
    LOG_ERR("Link failed for /s%u (pre/osd/post/cpu/enc/rtp/rtsp)", index);
    cleanup_branch(&be, enc_is_hw);
    g_mutex_unlock(&g_state_lock);
    return FALSE;
  }

  gchar *path = g_strdup_printf("/s%u", index);
  gchar *loc = NULL; if (strchr(g_push_url_tmpl, '%')) loc = g_strdup_printf(g_push_url_tmpl, index); else loc = g_strdup(g_push_url_tmpl);
  LOG_INF("Linked demux src_%u to rtspclientsink → %s", index, loc);
  if (out_path) *out_path = g_strdup(path);
  if (out_url) *out_url = loc; else g_free(loc);

  g_streams[index].in_use = TRUE;
  g_streams[index].enc_is_hw = enc_is_hw;
  strncpy(g_streams[index].enc_kind, enc_is_hw ? "nvenc" : (enc_is_x264 ? "x264" : (g_str_has_prefix(G_OBJECT_TYPE_NAME(be.enc), "GstAv") ? "avenc" : "openh264")), sizeof(g_streams[index].enc_kind)-1);
  g_streams[index].enc_kind[sizeof(g_streams[index].enc_kind)-1] = '\0';
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
  g_streams[index].rtsp = be.rtsp;

  g_free(path);
  g_mutex_unlock(&g_state_lock);
  return TRUE;
}
