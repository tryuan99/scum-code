#include <stdio.h>

#include "pico/cyw43_arch.h"
#include "pico/stdlib.h"
#include "pico/wifi/env.h"
#include "pico/wifi/http_client.h"

// WiFi connect timeout in milliseconds.
#define WIFI_CONNECT_TIMEOUT_MS 10000  // milliseconds

#define HOSTNAME "raw.githubusercontent.com"
#define URL "/tryuan99/scum-code/refs/heads/develop/MODULE.bazel"

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

  // Send an HTTP request.
  http_client_request_t request = {0};
  request.hostname = HOSTNAME;
  request.url = URL;
  request.headers_callback = http_client_headers_print_callback;
  request.receive_callback = http_client_receive_print_callback;
  const int error =
      http_client_request_sync(cyw43_arch_async_context(), &request);
  if (error != 0) {
    printf("HTTP request failed with error %d.", error);
  }

  cyw43_arch_deinit();
  return EXIT_SUCCESS;
}
