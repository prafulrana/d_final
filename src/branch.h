// Per-stream branch building
#ifndef BRANCH_H
#define BRANCH_H

#include <glib.h>

gboolean add_branch_and_mount(guint index, gchar **out_path, gchar **out_url);

#endif // BRANCH_H

