// App lifecycle and pipeline/RTSP setup
#ifndef APP_H
#define APP_H

#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include "config.h"

gboolean sanity_check_plugins(void);
void decide_max_streams(void);
gboolean build_full_pipeline_and_server(const AppConfig *cfg, GstPipeline **out_pipeline, GstRTSPServer **out_server);

gboolean app_setup(const AppConfig *cfg);
void app_loop(void);
void app_teardown(void);

#endif // APP_H

