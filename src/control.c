// REST API helpers for nvmultiurisrcbin (no custom control API)
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include "log.h"

// Minimal HTTP POST helper to nvmultiurisrcbin REST (localhost:9010/9000)
static gboolean http_post_localhost_port(const char *port_str, const char *path, const char *json, size_t json_len) {
  struct addrinfo hints; memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
  struct addrinfo *res = NULL;
  int err = getaddrinfo("127.0.0.1", port_str, &hints, &res);
  if (err != 0 || !res) { LOG_WRN("REST: getaddrinfo failed: %s", gai_strerror(err)); if (res) freeaddrinfo(res); return FALSE; }
  int s = -1; for (struct addrinfo *rp = res; rp != NULL; rp = rp->ai_next) {
    s = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (s == -1) continue;
    if (connect(s, rp->ai_addr, rp->ai_addrlen) == 0) break;
    close(s); s = -1;
  }
  freeaddrinfo(res);
  if (s == -1) { LOG_WRN("REST: connect failed"); return FALSE; }
  gchar *req = g_strdup_printf(
    "POST %s HTTP/1.1\r\nHost: 127.0.0.1:%s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
    path, port_str, json_len);
  ssize_t n = send(s, req, strlen(req), 0); g_free(req);
  if (n < 0) { LOG_WRN("REST: send header failed"); close(s); return FALSE; }
  if (json_len > 0) { ssize_t m = send(s, json, json_len, 0); if (m < 0) { LOG_WRN("REST: send body failed"); close(s); return FALSE; } }
  char buf[256]; (void)recv(s, buf, sizeof(buf), 0); close(s);
  return TRUE;
}

gboolean nvmulti_rest_post(const char *path, const char *json, size_t json_len) {
  if (http_post_localhost_port("9010", path, json, json_len)) return TRUE;
  if (http_post_localhost_port("9000", path, json, json_len)) return TRUE;
  LOG_WRN("REST: failed to post on 9010 and 9000");
  return FALSE;
}

gpointer control_http_thread(gpointer data) {
  (void)data;
  // No custom control API - only NVIDIA's nvmultiurisrcbin REST API
  return NULL;
}
