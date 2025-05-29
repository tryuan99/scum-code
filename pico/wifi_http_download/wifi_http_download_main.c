#include <stdio.h>
#include <stdlib.h>

#include "lwip/altcp.h"
#include "lwip/apps/http_client.h"
#include "pico/cyw43_arch.h"
#include "pico/stdio.h"
#include "pico/wifi/env.h"
#include "pico/wifi/http_client.h"

// WiFi connect timeout in milliseconds.
#define WIFI_CONNECT_TIMEOUT_MS 10000  // milliseconds

// WiFi maximum size of the HTTP body in bytes.
#define WIFI_MAX_HTTP_BODY_SIZE (1 << 16)  // bytes

#define HOSTNAME "people.eecs.berkeley.edu"
#define URL "/~titan/hello_world.bin"

// HTTP body buffer.
uint8_t g_http_body_buffer[WIFI_MAX_HTTP_BODY_SIZE];

// HTTP body buffer size.
size_t g_http_body_buffer_size = 0;

// HTTP receive callback function that saves the HTTP body.
err_t http_receive_save_callback(void* arg, struct altcp_pcb* connection,
                                 struct pbuf* packet, const err_t error) {
  for (size_t i = 0;
       i < packet->tot_len && g_http_body_buffer_size < WIFI_MAX_HTTP_BODY_SIZE;
       ++i, ++g_http_body_buffer_size) {
    g_http_body_buffer[g_http_body_buffer_size] = pbuf_get_at(packet, i);
  }
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
  request.receive_callback = http_receive_save_callback;
  const int error =
      http_client_request_sync(cyw43_arch_async_context(), &request);
  printf("Downloaded %u bytes in the HTML body.\n", g_http_body_buffer_size);
  if (error != 0) {
    printf("HTTP request failed with error %d.\n", error);
  }

  cyw43_arch_deinit();
  return EXIT_SUCCESS;
}
