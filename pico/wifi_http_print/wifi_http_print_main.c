#include <stdio.h>
#include <stdlib.h>

#include "lwip/altcp.h"
#include "lwip/apps/http_client.h"
#include "pico/cyw43_arch.h"
#include "pico/stdio.h"
#include "pico/wifi/env.h"
#include "pico/wifi/http_client.h"

// WiFi connect timeout in milliseconds.
#define WIFI_CONNECT_TIMEOUT_MS 10000

#define HOSTNAME "people.eecs.berkeley.edu"
#define URL "/~titan/hello_world.bin"

// HTTP body size.
static size_t g_http_body_size = 0;

// HTTP headers callback function that prints the headers.
static err_t http_headers_print_callback(httpc_state_t* connection, void* arg,
                                         struct pbuf* headers,
                                         const uint16_t headers_length,
                                         const uint32_t content_length) {
  printf("Headers:\n");
  for (size_t i = 0; i < headers->tot_len && i < headers_length; ++i) {
    printf("%c", pbuf_get_at(headers, i));
  }
  return ERR_OK;
}

// HTTP receive callback function that prints the HTTP body.
static err_t http_receive_print_callback(void* arg,
                                         struct altcp_pcb* connection,
                                         struct pbuf* packet,
                                         const err_t error) {
  printf("Body:\n");
  for (size_t i = 0; i < packet->tot_len; ++i) {
    printf("%c", pbuf_get_at(packet, i));
  }
  g_http_body_size += packet->tot_len;
  printf("\n");
  return ERR_OK;
}

int main(int argc, char** argv) {
  stdio_init_all();
  cyw43_arch_init();
  cyw43_arch_enable_sta_mode();

  // Connect to the WiFi network.
  if (cyw43_arch_wifi_connect_timeout_ms(
          WIFI_SSID, WIFI_PASSWORD, WIFI_ENCRYPTION, WIFI_CONNECT_TIMEOUT_MS)) {
    printf("Failed to connect to WiFi.\n");
    return EXIT_FAILURE;
  }
  printf("Connected to WiFi network %s.\n", WIFI_SSID);

  // Send the HTTP request.
  http_client_request_t request = {0};
  request.hostname = HOSTNAME;
  request.url = URL;
  request.headers_callback = http_headers_print_callback;
  request.receive_callback = http_receive_print_callback;
  const int error =
      http_client_request_sync(cyw43_arch_async_context(), &request);
  printf("Received %u bytes in the HTML body.\n", g_http_body_size);
  if (error != 0) {
    printf("HTTP request failed with error %d.\n", error);
  }

  cyw43_arch_deinit();
  return EXIT_SUCCESS;
}
