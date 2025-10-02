// Shared state and per-stream metadata
#ifndef STATE_H
#define STATE_H

#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib.h>

typedef struct {
  gboolean in_use;
  gboolean enc_is_hw;
  char enc_kind[16]; // nvenc, x264, avenc, openh264
  guint udp_port;
  char path[16];     // /sN
  GstElement *queue;
  GstElement *conv_pre;
  GstElement *caps_pre;
  GstElement *osd;
  GstElement *conv_post;
  GstElement *caps_post;
  GstElement *conv_cpu;
  GstElement *caps_cpu;
  GstElement *enc;
  GstElement *parse;
  GstElement *pay;
  GstElement *udp;
} StreamInfo;

extern GstPipeline *g_pipeline;
extern GstElement  *g_pre_bin;
extern GstElement  *g_demux;
extern GstRTSPServer *g_rtsp_server;
extern guint g_rtsp_port;
extern guint g_next_index;
extern guint g_base_udp_port_glb;
extern GMutex g_state_lock;
extern guint g_ctrl_port;
extern const gchar *g_public_host;
extern guint g_hw_threshold;  // NVENC for first N
extern guint g_sw_max;        // SW encoder count
extern guint g_max_streams;   // total allowed
extern StreamInfo g_streams[64];

#endif // STATE_H

