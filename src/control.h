// REST API helpers for nvmultiurisrcbin (no custom control API)
#ifndef CONTROL_H
#define CONTROL_H

#include <glib.h>

gpointer control_http_thread(gpointer data);

// REST API helpers for nvmultiurisrcbin
gboolean nvmulti_rest_post(const char *path, const char *json, size_t json_len);

#endif // CONTROL_H
