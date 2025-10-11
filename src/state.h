// Shared state and per-stream metadata
#ifndef STATE_H
#define STATE_H

#include <gst/gst.h>
#include <glib.h>

typedef struct {
  gboolean in_use;
  gboolean enc_is_hw;
  gboolean eos;         // TRUE if stream has EOS'd (waiting for reconnect)
  guint reconnect_count; // Number of EOS events for this stream
  char enc_kind[16]; // nvenc, x264, avenc, openh264
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
  // Direct push sink
  GstElement *rtsp;
} StreamInfo;

extern GstPipeline *g_pipeline;
extern GstElement  *g_pre_bin;
extern GstElement  *g_demux;
extern guint g_next_index;
extern GMutex g_state_lock;
extern guint g_ctrl_port;
extern const gchar *g_push_url_tmpl; // e.g., rtsp://host:8554/s%u
// Optional: when set, push directly to remote RTSP using rtspclientsink.
extern guint g_hw_threshold;  // NVENC for first N
extern guint g_sw_max;        // SW encoder count
extern guint g_max_streams;   // total allowed
extern StreamInfo g_streams[64];

#endif // STATE_H
