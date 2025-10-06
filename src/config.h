// App configuration (read from CLI/env)
#ifndef CONFIG_H
#define CONFIG_H

#include <glib.h>

typedef struct {
  // Outputs / serving
  guint rtsp_port;     // RTSP TCP port (e.g., 8554)
  guint base_udp_port; // base UDP port for per-stream RTP egress

  // Sample source + URL host for responses
  gchar *sample_uri;  // default DS sample video
  gchar *public_host; // host/IP for returned RTSP URLs
} AppConfig;

gboolean parse_args(int argc, char *argv[], AppConfig *cfg);
void cleanup_config(AppConfig *cfg);

#endif // CONFIG_H
