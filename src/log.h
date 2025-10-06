// Simple logging macros for consistent, readable output
#ifndef LOG_H
#define LOG_H

#include <glib.h>

#define LOG_ERR(fmt, ...) g_printerr("ERR: " fmt "\n", ##__VA_ARGS__)
#define LOG_WRN(fmt, ...) g_printerr("WRN: " fmt "\n", ##__VA_ARGS__)
#define LOG_INF(fmt, ...) g_print("INF: " fmt "\n", ##__VA_ARGS__)

#endif // LOG_H

