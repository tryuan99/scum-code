#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lwip/altcp.h"
#include "lwip/apps/http_client.h"
#include "pico/cyw43_arch.h"
#include "pico/defs.h"
#include "pico/error.h"
#include "pico/scum/bootloader.h"
#include "pico/scum/uart.h"
#include "pico/stdio.h"
#include "pico/stdio_usb.h"
#include "pico/time.h"
#include "pico/wifi/env.h"
#include "pico/wifi/http_client.h"

// WiFi connect timeout in milliseconds.
#define WIFI_CONNECT_TIMEOUT_MS 10000

// SCuM binary last modified time size in bytes.
#define SCUM_BINARY_LAST_MODIFIED_SIZE 64

// SCuM binary last modified time timer period in milliseconds.
#define SCUM_BINARY_LAST_MODIFIED_PERIOD_MS 10000

// Hostname and URL of the SCuM binary.
#define SCUM_BINARY_HOSTNAME "people.eecs.berkeley.edu"
#define SCUM_BINARY_URL "/~titan/hello_world.bin"

// SCuM WiFi bootloading state enumeration.
typedef enum {
  STATE_INVALID = -1,
  STATE_INIT,
  STATE_IDLE,
  STATE_CHECK_FOR_NEW_BINARY,
  STATE_DOWNLOAD_BINARY,
  STATE_BOOTLOAD,
  STATE_DONE,
} scum_bootloader_wifi_state_e;

// SCuM bootloading state.
static scum_bootloader_wifi_state_e g_scum_bootloader_wifi_state = STATE_INIT;

// SCuM binary.
static scum_binary_t g_scum_bootloader_wifi_binary;

// SCuM binary size in bytes.
static size_t g_scum_bootloader_wifi_binary_size = 0;

// Last modified time of the SCuM binary.
static char
    g_scum_bootloader_wifi_binary_last_modified[SCUM_BINARY_LAST_MODIFIED_SIZE];

// Number of bytes in the last modified time.
static size_t g_scum_bootloader_wifi_binary_last_modified_size = 0;

// Last modified time timer.
static repeating_timer_t g_scum_bootloader_wifi_binary_last_modified_timer;

// SCuM binary last modified time timer callback.
static bool scum_binary_last_modified_timer_callback(repeating_timer_t* timer) {
  if (g_scum_bootloader_wifi_state == STATE_IDLE) {
    g_scum_bootloader_wifi_state = STATE_CHECK_FOR_NEW_BINARY;
  }
  return true;
}

// Save the last modified time of the SCuM binary within the HTTP headers.
static err_t scum_bootloader_wifi_http_receive_headers_callback(
    httpc_state_t* connection, void* arg, struct pbuf* headers,
    const uint16_t headers_length, const uint32_t content_length) {
  // Search for the last modified time within the HTTP headers.
  const char* last_modified = "last-modified: ";
  const uint16_t last_modified_position =
      pbuf_memfind(headers, last_modified, strlen(last_modified), /*offset=*/0);
  const char* newline = "\n";
  const uint16_t newline_position =
      pbuf_memfind(headers, newline, strlen(newline), last_modified_position);
  memset(g_scum_bootloader_wifi_binary_last_modified, 0,
         sizeof(char) * SCUM_BINARY_LAST_MODIFIED_SIZE);
  g_scum_bootloader_wifi_binary_last_modified_size = pbuf_copy_partial(
      headers, g_scum_bootloader_wifi_binary_last_modified,
      /*len=*/newline_position - last_modified_position - strlen(last_modified),
      /*offset=*/last_modified_position + strlen(last_modified));
  return ERR_OK;
}

// Save the SCuM binary received within the HTTP body.
static err_t scum_bootloader_wifi_http_receive_binary_callback(
    void* arg, struct altcp_pcb* connection, struct pbuf* packet,
    const err_t error) {
  const uint16_t size =
      (g_scum_bootloader_wifi_binary_size + packet->tot_len > SCUM_BINARY_SIZE)
          ? (SCUM_BINARY_SIZE - g_scum_bootloader_wifi_binary_size)
          : packet->tot_len;
  g_scum_bootloader_wifi_binary_size += pbuf_copy_partial(
      packet,
      &g_scum_bootloader_wifi_binary.data[g_scum_bootloader_wifi_binary_size],
      size,
      /*offset=*/0);
  return ERR_OK;
}

// Return whether a new SCuM binary exists.
static bool scum_bootloader_wifi_has_new_binary(void) {
  // Copy the last modified time of the SCuM binary.
  char old_binary_last_modified[SCUM_BINARY_LAST_MODIFIED_SIZE];
  memset(old_binary_last_modified, 0,
         sizeof(char) * SCUM_BINARY_LAST_MODIFIED_SIZE);
  memcpy(old_binary_last_modified, g_scum_bootloader_wifi_binary_last_modified,
         g_scum_bootloader_wifi_binary_last_modified_size);

  // Check the last modified time of the SCuM binary.
  http_client_request_t request = {0};
  request.hostname = SCUM_BINARY_HOSTNAME;
  request.url = SCUM_BINARY_URL;
  request.headers_callback = scum_bootloader_wifi_http_receive_headers_callback;
  const int error =
      http_client_request_sync(cyw43_arch_async_context(), &request);
  if (error != 0) {
    printf("HTTP request failed with error %d.\n", error);
    return false;
  }
  if (g_scum_bootloader_wifi_binary_last_modified_size == 0) {
    return true;
  }
  return memcmp(g_scum_bootloader_wifi_binary_last_modified,
                old_binary_last_modified,
                g_scum_bootloader_wifi_binary_last_modified_size) != 0;
}

// SCuM bootloading complete callback function.
static void scum_bootloader_wifi_complete_callback(void) {
  g_scum_bootloader_wifi_state = STATE_DONE;
}

int main(int argc, char** argv) {
  // Initialize USB.
  stdio_usb_init();

  // Limit input and output to USB only.
  stdio_filter_driver(&stdio_usb);

  // Initialize SCuM's UART.
  scum_uart_init();

  // Initialize the SCuM bootloader.
  scum_bootloader_init();

  // Initialize the WiFi.
  cyw43_arch_init();
  cyw43_arch_enable_sta_mode();

  // Connect to the WiFi network.
  int error = cyw43_arch_wifi_connect_timeout_ms(
      WIFI_SSID, WIFI_PASSWORD, WIFI_ENCRYPTION, WIFI_CONNECT_TIMEOUT_MS);
  while (error != PICO_OK) {
    printf("Failed to connect to WiFi with error %d.\n", error);
    error = cyw43_arch_wifi_connect_timeout_ms(
        WIFI_SSID, WIFI_PASSWORD, WIFI_ENCRYPTION, WIFI_CONNECT_TIMEOUT_MS);
  }
  printf("Connected to WiFi network %s.\n", WIFI_SSID);

  if (!add_repeating_timer_ms(
          SCUM_BINARY_LAST_MODIFIED_PERIOD_MS,
          scum_binary_last_modified_timer_callback,
          /*user_data=*/NULL,
          &g_scum_bootloader_wifi_binary_last_modified_timer)) {
    printf("Failed to create the SCuM binary last modified time timer.\n");
    return EXIT_FAILURE;
  }

  g_scum_bootloader_wifi_state = STATE_IDLE;
  while (true) {
    switch (g_scum_bootloader_wifi_state) {
      case STATE_IDLE: {
        scum_uart_print();
        break;
      }
      case STATE_CHECK_FOR_NEW_BINARY: {
        // Check if a new SCuM binary exists.
        if (scum_bootloader_wifi_has_new_binary()) {
          printf("Found new SCuM binary with last modified time of %s.\n",
                 g_scum_bootloader_wifi_binary_last_modified);
          scum_uart_stop();
          g_scum_bootloader_wifi_state = STATE_DOWNLOAD_BINARY;
        } else {
          g_scum_bootloader_wifi_state = STATE_IDLE;
        }
        break;
      }
      case STATE_DOWNLOAD_BINARY: {
        printf("Downloading SCuM binary http://%s%s.\n", SCUM_BINARY_HOSTNAME,
               SCUM_BINARY_URL);
        memset(&g_scum_bootloader_wifi_binary, 0, sizeof(scum_binary_t));
        g_scum_bootloader_wifi_binary_size = 0;

        // Send the HTTP request.
        http_client_request_t request = {0};
        request.hostname = SCUM_BINARY_HOSTNAME;
        request.url = SCUM_BINARY_URL;
        request.receive_callback =
            scum_bootloader_wifi_http_receive_binary_callback;
        const int error =
            http_client_request_sync(cyw43_arch_async_context(), &request);
        printf("Downloaded %u bytes for the SCuM binary.\n",
               g_scum_bootloader_wifi_binary_size);
        if (error != 0) {
          printf("HTTP request failed with error %d.\n", error);
        } else {
          scum_bootloader_run(&g_scum_bootloader_wifi_binary,
                              scum_bootloader_wifi_complete_callback);
          g_scum_bootloader_wifi_state = STATE_BOOTLOAD;
        }
        break;
      }
      case STATE_BOOTLOAD: {
        scum_bootloader_loop();
        break;
      }
      case STATE_DONE: {
        scum_uart_start();
        g_scum_bootloader_wifi_state = STATE_IDLE;
        break;
      }
      default: {
        break;
      }
    }
  }

  cyw43_arch_deinit();
  return EXIT_SUCCESS;
}
