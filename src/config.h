// App configuration (read from CLI/env)
#ifndef CONFIG_H
#define CONFIG_H

#include <glib.h>

typedef struct {
  // Sample source
  gchar *sample_uri;  // default DS sample video
  // Optional: when set, bypass local RTSP server and push directly to remote via rtspclientsink.
  // Example: rtsp://34.93.89.70:8554/s%u
  gchar *push_url_tmpl;
} AppConfig;

gboolean parse_args(int argc, char *argv[], AppConfig *cfg);
void cleanup_config(AppConfig *cfg);

#endif // CONFIG_H
