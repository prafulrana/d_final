// App lifecycle and pipeline/RTSP setup
#ifndef APP_H
#define APP_H

#include <gst/gst.h>
#include "config.h"

gboolean sanity_check_plugins(void);
void decide_max_streams(void);
gboolean build_pipeline(const AppConfig *cfg, GstPipeline **out_pipeline);

gboolean app_setup(const AppConfig *cfg);
void app_loop(void);
void app_teardown(void);

#endif // APP_H
