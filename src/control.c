// Tiny control HTTP server (L2)
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include "log.h"
#include "state.h"
#include "branch.h"

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

static gboolean http_post_localhost_try(const char *path, const char *json, size_t json_len) {
  if (http_post_localhost_port("9010", path, json, json_len)) return TRUE;
  if (http_post_localhost_port("9000", path, json, json_len)) return TRUE;
  LOG_WRN("REST: failed to post on 9010 and 9000");
  return FALSE;
}

static int create_ctrl_listener(void) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) { LOG_ERR("CTRL: socket create failed"); return -1; }
  int opt = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  struct sockaddr_in addr; memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_ANY);
  const gchar *env = g_getenv("CTRL_PORT");
  if (env && *env) {
    guint port = (guint) g_ascii_strtoull(env, NULL, 10);
    addr.sin_port = htons((uint16_t)port);
    if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) == 0) { g_ctrl_port = port; }
    else { LOG_WRN("CTRL: bind %u from env failed; falling back to 8080-8089", port); }
  }
  if (g_ctrl_port == 0) {
    for (int port = 8080; port < 8090; ++port) {
      addr.sin_port = htons((uint16_t)port);
      if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) == 0) { g_ctrl_port = (guint)port; break; }
    }
  }
  if (g_ctrl_port == 0) { LOG_ERR("CTRL: bind 8080-8089 failed"); close(s); return -1; }
  if (listen(s, 16) != 0) { LOG_ERR("CTRL: listen failed"); close(s); return -1; }
  LOG_INF("Control API listening on http://0.0.0.0:%u", g_ctrl_port);
  return s;
}

static void handle_request(int c, const char *buf) {
  gboolean is_add = (g_str_has_prefix(buf, "GET /add_demo_stream ") || g_str_has_prefix(buf, "GET /add_demo_stream?"));
  gboolean is_status = (g_str_has_prefix(buf, "GET /status ") || g_str_has_prefix(buf, "GET /status?"));
  if (!is_add && !is_status) {
    const char *resp = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found";
    send(c, resp, strlen(resp), 0);
    return;
  }
  if (is_status) {
    GString *j = g_string_new(NULL);
    g_string_append_printf(j, "{\n  \"max\": %u,\n  \"streams\": [\n", g_max_streams);
    gboolean first = TRUE;
    for (guint i = 0; i < G_N_ELEMENTS(g_streams); ++i) {
      if (!g_streams[i].in_use) continue;
      if (!first) g_string_append(j, ",\n");
      first = FALSE;
      g_string_append_printf(j,
        "    { \"index\": %u, \"path\": \"%s\", \"udp\": %u, \"encoder\": \"%s\" }",
        i, g_streams[i].path, g_streams[i].udp_port, g_streams[i].enc_kind[0] ? g_streams[i].enc_kind : "unknown");
    }
    g_string_append(j, "\n  ]\n}\n");
    gchar *hdr = g_strdup_printf("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", j->len);
    send(c, hdr, strlen(hdr), 0); send(c, j->str, j->len, 0);
    g_free(hdr); g_string_free(j, TRUE);
    return;
  }

  // Add stream
  guint index;
  g_mutex_lock(&g_state_lock);
  if (g_next_index >= g_max_streams) {
    g_mutex_unlock(&g_state_lock);
    gchar *json = g_strdup_printf("{\n  \"error\": \"capacity_exceeded\",\n  \"max\": %u\n}\n", g_max_streams);
    gchar *hdr = g_strdup_printf("HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", strlen(json));
    send(c, hdr, strlen(hdr), 0); send(c, json, strlen(json), 0);
    g_free(hdr); g_free(json);
    return;
  }
  index = g_next_index++;
  g_mutex_unlock(&g_state_lock);

  gchar *path = NULL; gchar *url = NULL;
  gboolean ok = add_branch_and_mount(index, &path, &url);
  if (ok) {
    const gchar *env_sample = g_getenv("SAMPLE_URI");
    const gchar *sample_uri = env_sample ? env_sample : "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4";
    gchar *body = g_strdup_printf(
      "{\n  \"key\": \"sensor\",\n  \"value\": {\n    \"camera_id\": \"api_%u\",\n    \"camera_url\": \"%s\",\n    \"change\": \"camera_add\"\n  },\n  \"headers\": { \"source\": \"app\" }\n}\n",
      index, sample_uri);
    (void)http_post_localhost_try("/api/v1/stream/add", body, strlen(body));
    g_free(body);
    gchar *json = g_strdup_printf("{\n  \"path\": \"%s\",\n  \"url\": \"%s\"\n}\n", path, url);
    gchar *hdr = g_strdup_printf("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", strlen(json));
    send(c, hdr, strlen(hdr), 0); send(c, json, strlen(json), 0);
    g_free(hdr); g_free(json);
  } else {
    const char *resp = "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: 6\r\nConnection: close\r\n\r\nError\n";
    send(c, resp, strlen(resp), 0);
  }
  g_free(path); g_free(url);
}

gpointer control_http_thread(gpointer data) {
  (void)data;
  int s = create_ctrl_listener();
  if (s < 0) return NULL;
  for (;;) {
    int c = accept(s, NULL, NULL);
    if (c < 0) continue;
    char buf[1024]; ssize_t n = recv(c, buf, sizeof(buf)-1, 0);
    if (n <= 0) { close(c); continue; }
    buf[n] = '\0';
    handle_request(c, buf);
    close(c);
  }
  return NULL;
}

