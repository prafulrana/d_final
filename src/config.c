// Config parsing and helpers
#include "config.h"
#include <glib.h>
#include <stdio.h>

gboolean parse_args(int argc, char *argv[], AppConfig *cfg) {
  // Defaults
  cfg->sample_uri = g_strdup("file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4");
  cfg->push_url_tmpl = NULL;

  // CLI (optional; first arg can be a file path for compatibility)
  (void)argc; (void)argv; // no file-based pipeline
  (void)argc; (void)argv;

  // Environment overrides
  const gchar *env;
  if ((env = g_getenv("SAMPLE_URI"))) { g_free(cfg->sample_uri); cfg->sample_uri = g_strdup(env); }
  if ((env = g_getenv("RTSP_PUSH_URL_TMPL"))) { cfg->push_url_tmpl = g_strdup(env); }
  return TRUE;
}

void cleanup_config(AppConfig *cfg) {
  g_free(cfg->sample_uri);
  g_free(cfg->push_url_tmpl);
}
