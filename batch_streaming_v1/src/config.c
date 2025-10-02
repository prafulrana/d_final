// Config parsing and helpers
#include "config.h"
#include <glib.h>
#include <stdio.h>

gboolean parse_args(int argc, char *argv[], AppConfig *cfg) {
  // Defaults
  cfg->rtsp_port = 8554;
  cfg->base_udp_port = 5000;
  cfg->sample_uri = g_strdup("file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4");
  cfg->public_host = g_strdup("127.0.0.1");

  // CLI (optional; first arg can be a file path for compatibility)
  (void)argc; (void)argv; // no file-based pipeline
  if (argc >= 3) cfg->rtsp_port = (guint) g_ascii_strtoull(argv[2], NULL, 10);

  // Environment overrides
  const gchar *env;
  if ((env = g_getenv("RTSP_PORT"))) cfg->rtsp_port = (guint) g_ascii_strtoull(env, NULL, 10);
  if ((env = g_getenv("BASE_UDP_PORT"))) cfg->base_udp_port = (guint) g_ascii_strtoull(env, NULL, 10);
  if ((env = g_getenv("SAMPLE_URI"))) { g_free(cfg->sample_uri); cfg->sample_uri = g_strdup(env); }
  if ((env = g_getenv("PUBLIC_HOST"))) { g_free(cfg->public_host); cfg->public_host = g_strdup(env); }
  return TRUE;
}

void cleanup_config(AppConfig *cfg) {
  g_free(cfg->sample_uri);
  g_free(cfg->public_host);
}
