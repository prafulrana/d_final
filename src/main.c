//
// Simple DeepStream Pipeline (Beginner Friendly)
// ---------------------------------------------
// Read this first:
// - main() calls a few small functions so it’s easy to follow.
// - The details live in app.c (setup/loop/teardown), branch.c (per-stream),
//   control.c (tiny HTTP control), and config.c (args/env).
// - Start with app_setup() to see how the pipeline is built and pushed.

#include <gst/gst.h>
#include "app.h"
#include "config.h"
#include "log.h"

int main(int argc, char *argv[]) {
  AppConfig cfg;
  if (!parse_args(argc, argv, &cfg)) {
    LOG_ERR("Argument parsing failed");
    return 1;
  }

  gst_init(&argc, &argv);

  if (!sanity_check_plugins()) {
    cleanup_config(&cfg);
    return 3;
  }

  if (!app_setup(&cfg)) {
    cleanup_config(&cfg);
    return 2;
  }

  app_loop();
  app_teardown();
  cleanup_config(&cfg);
  return 0;
}
