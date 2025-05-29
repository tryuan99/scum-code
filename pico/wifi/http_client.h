// The HTTP module manages HTTP requests and responses.

#ifndef PICO_WIFI_HTTP_H_
#define PICO_WIFI_HTTP_H_

#include <stdbool.h>
#include <stdint.h>

#include "lwip/altcp.h"
#include "lwip/apps/http_client.h"
#include "pico/async_context.h"

typedef struct {
  // Host name, e.g., www.raspberrypi.com.
  const char* hostname;

  // The URL to request, e.g., /favicon.ico.
  const char* url;

  // Headers callback function.
  httpc_headers_done_fn headers_callback;

  // Receive callback function.
  altcp_recv_fn receive_callback;

  // Result callback function.
  httpc_result_fn result_callback;

  // Callback argument.
  void* callback_arg;

  // The port to use. If zero, a default port will be used.
  uint16_t port;

  // HTTP client settings.
  httpc_connection_t settings;

  // If true, the HTTP request is complete.
  bool complete;

  // HTTP result.
  httpc_result_t result;
} http_client_request_t;

// Perform an HTTP request asynchronously and return once the request has
// beenmade. The request is complete when request->complete is true or when the
// result callback function has been called. If success, return 0.
int http_client_request_async(async_context_t* context,
                              http_client_request_t* request);

// Perform an HTTP request synchronously and return once the request is
// complete. If success, return 0.
int http_client_request_sync(async_context_t* context,
                             http_client_request_t* request);

#endif  // #ifndef PICO_WIFI_HTTP_H_
