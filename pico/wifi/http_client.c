#include "pico/wifi/http_client.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "lwip/altcp.h"
#include "lwip/apps/http_client.h"
#include "pico/async_context.h"

// Default HTTP port.
#define HTTP_PORT 80

// HTTP response poll period in milliseconds.
#define HTTP_CLIENT_RESPONSE_POLL_PERIOD_MS 100

// Wrapper for the HTTP headers callback function.
static err_t http_client_headers_callback_wrapper(
    httpc_state_t* connection, void* arg, struct pbuf* headers,
    const uint16_t headers_length, const uint32_t content_length) {
  http_client_request_t* request = (http_client_request_t*)arg;
  if (request->headers_callback) {
    return request->headers_callback(connection, request->callback_arg, headers,
                                     headers_length, content_length);
  }
  return ERR_OK;
}

// Wrapper for the HTTP receive callback function.
static err_t http_client_receive_callback_wrapper(void* arg,
                                                  struct altcp_pcb* connection,
                                                  struct pbuf* packet,
                                                  const err_t error) {
  http_client_request_t* request = (http_client_request_t*)arg;
  if (request->receive_callback) {
    return request->receive_callback(request->callback_arg, connection, packet,
                                     error);
  }
  return ERR_OK;
}

// Wrapper for the HTTP result callback function.
static void http_client_result_callback_wrapper(
    void* arg, const httpc_result_t result, const uint32_t rx_content_length,
    const uint32_t response, const err_t error) {
  http_client_request_t* request = (http_client_request_t*)arg;
  request->complete = true;
  request->result = result;
  if (request->result_callback) {
    request->result_callback(request->callback_arg, result, rx_content_length,
                             response, error);
  }
}

int http_client_request_async(async_context_t* context,
                              http_client_request_t* request) {
  request->complete = false;
  request->settings.headers_done_fn = (request->headers_callback != NULL)
                                          ? http_client_headers_callback_wrapper
                                          : NULL;
  request->settings.result_fn = http_client_result_callback_wrapper;
  async_context_acquire_lock_blocking(context);
  const uint16_t port = (request->port != 0) ? request->port : HTTP_PORT;
  const err_t error = httpc_get_file_dns(
      request->hostname, port, request->url, &request->settings,
      http_client_receive_callback_wrapper, request, /*connection=*/NULL);
  async_context_release_lock(context);
  if (error != ERR_OK) {
    printf("HTTP request failed with error %d.", error);
  }
  return (int)error;
}

int http_client_request_sync(async_context_t* context,
                             http_client_request_t* request) {
  const int error = http_client_request_async(context, request);
  if (error != 0) {
    return error;
  }
  while (!request->complete) {
    async_context_poll(context);
    async_context_wait_for_work_ms(context,
                                   HTTP_CLIENT_RESPONSE_POLL_PERIOD_MS);
  }
  return (int)request->result;
}
