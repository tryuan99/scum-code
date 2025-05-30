#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hardware/gpio.h"
#include "hardware/uart.h"
#include "lwip/altcp.h"
#include "lwip/apps/http_client.h"
#include "pico/cyw43_arch.h"
#include "pico/defs.h"
#include "pico/error.h"
#include "pico/multicore.h"
#include "pico/stdio.h"
#include "pico/stdio_uart.h"
#include "pico/stdio_usb.h"
#include "pico/time.h"
#include "pico/util/queue.h"
#include "pico/wifi/env.h"
#include "pico/wifi/http_client.h"

// WiFi connect timeout in milliseconds.
#define WIFI_CONNECT_TIMEOUT_MS 10000

// SCuM binary size in bytes.
// SCuM has a program memory size of 64 KiB.
#define SCUM_BINARY_SIZE (1 << 16)

// SCuM binary last modified time size in bytes.
#define SCUM_BINARY_LAST_MODIFIED_SIZE 64

// SCuM binary last modified time timer period in milliseconds.
#define SCUM_BINARY_LAST_MODIFIED_PERIOD_MS 10000

// Hostname and URL of the SCuM binary.
#define SCUM_BINARY_HOSTNAME "people.eecs.berkeley.edu"
#define SCUM_BINARY_URL "/~titan/hello_world.bin"

// 3WB pins to SCuM.
#define SCUM_CLOCK_PIN 2
#define SCUM_DATA_PIN 3
#define SCUM_ENABLE_PIN 4
#define SCUM_HRESET_PIN 5

// UART configuration for SCuM.
#define SCUM_UART_INSTANCE uart0
#define SCUM_UART_RX_PIN 1
#define SCUM_UART_BAUD_RATE 19200
#define SCUM_UART_BUFFER_SIZE 512

// Hard reset sleep time in microseconds.
#define SCUM_HARD_RESET_SLEEP_TIME_US 15000

// Toggle sleep time in microseconds.
#define SCUM_TOGGLE_SLEEP_TIME_US 1

// Calibration number of pulses.
#define SCUM_CALIBRATION_NUM_PULSES 30

// Calibration clock period in milliseconds.
#define SCUM_CALIBRATION_CLOCK_PERIOD_MS 100

// Calibration clock active time in microseconds.
#define SCUM_CALIBRATION_CLOCK_ACTIVE_TIME_US 20

// SCuM bootloading state enumeration.
typedef enum {
  STATE_INVALID = -1,
  STATE_INIT,
  STATE_IDLE,
  STATE_CHECK_FOR_NEW_BINARY,
  STATE_DOWNLOAD_BINARY,
  STATE_HARD_RESET,
  STATE_WRITE_BINARY,
  STATE_CALIBRATION,
  STATE_UART_MONITOR,
} scum_bootloader_state_e;

// SCuM bootloading state.
static scum_bootloader_state_e g_scum_bootloader_state = STATE_INIT;

// SCuM binary.
static uint8_t g_scum_bootloader_binary[SCUM_BINARY_SIZE];

// Number of bytes received.
static size_t g_scum_bootloader_binary_size = 0;

// Last modified time of the SCuM binary.
static char
    g_scum_bootloader_binary_last_modified[SCUM_BINARY_LAST_MODIFIED_SIZE];

// Number of bytes in the last modified time.
static size_t g_scum_bootloader_binary_last_modified_size = 0;

// Last modified time timer.
static repeating_timer_t g_scum_bootloader_binary_last_modified_timer;

// Calibration timer.
static repeating_timer_t g_scum_calibration_timer;

// Calibration number of pulses.
static uint32_t g_scum_calibration_num_pulses = 0;

// UART buffer.
static queue_t g_scum_uart_buffer;

// Initialize the GPIOs.
static inline void scum_bootloader_gpio_init() {
  gpio_init(SCUM_CLOCK_PIN);
  gpio_set_dir(SCUM_CLOCK_PIN, GPIO_OUT);
  gpio_init(SCUM_DATA_PIN);
  gpio_set_dir(SCUM_DATA_PIN, GPIO_OUT);
  gpio_init(SCUM_ENABLE_PIN);
  gpio_set_dir(SCUM_ENABLE_PIN, GPIO_OUT);

  // The hard reset pin is set to high-Z.
  gpio_init(SCUM_HRESET_PIN);
  gpio_set_dir(SCUM_HRESET_PIN, GPIO_IN);

  // Disable all pull-up and pull-down resistors.
  gpio_disable_pulls(SCUM_CLOCK_PIN);
  gpio_disable_pulls(SCUM_DATA_PIN);
  gpio_disable_pulls(SCUM_ENABLE_PIN);
  gpio_disable_pulls(SCUM_HRESET_PIN);

  // Decrease the drive strengths of the GPIOs.
  gpio_set_drive_strength(SCUM_CLOCK_PIN, GPIO_DRIVE_STRENGTH_2MA);
  gpio_set_drive_strength(SCUM_DATA_PIN, GPIO_DRIVE_STRENGTH_2MA);
  gpio_set_drive_strength(SCUM_ENABLE_PIN, GPIO_DRIVE_STRENGTH_2MA);
  gpio_set_drive_strength(SCUM_HRESET_PIN, GPIO_DRIVE_STRENGTH_2MA);
}

// Initialize the LED.
static inline void scum_bootloader_led_init() {
  gpio_init(PICO_LED_PIN);
  gpio_set_dir(PICO_LED_PIN, GPIO_OUT);
}

// Calibration active time alarm callback.
static int64_t scum_calibration_active_time_alarm_callback(const alarm_id_t id,
                                                           void* user_data) {
  gpio_put(SCUM_CLOCK_PIN, false);
  return 0;
}

// Calibration timer callback.
static bool scum_calibration_timer_callback(repeating_timer_t* timer) {
  gpio_put(SCUM_CLOCK_PIN, true);
  add_alarm_in_us(SCUM_CALIBRATION_CLOCK_ACTIVE_TIME_US,
                  scum_calibration_active_time_alarm_callback,
                  /*user_data=*/NULL, /*fire_if_past=*/true);
  ++g_scum_calibration_num_pulses;
  return g_scum_calibration_num_pulses < SCUM_CALIBRATION_NUM_PULSES;
}

// SCuM UART reader.
static void scum_uart_reader() {
  while (true) {
    if (uart_is_readable(SCUM_UART_INSTANCE)) {
      char data = uart_getc(SCUM_UART_INSTANCE);
      queue_try_add(&g_scum_uart_buffer, &data);
    }
  }
}

// SCuM binary last modified time timer callback.
static bool scum_binary_last_modified_timer_callback(repeating_timer_t* timer) {
  if (g_scum_bootloader_state == STATE_IDLE) {
    g_scum_bootloader_state = STATE_CHECK_FOR_NEW_BINARY;
  }
  return true;
}

// Save the last modified time of the SCuM binary within the HTTP headers.
static err_t scum_bootloader_http_receive_headers_callback(
    httpc_state_t* connection, void* arg, struct pbuf* headers,
    const uint16_t headers_length, const uint32_t content_length) {
  // Search for the last modified time within the HTTP headers.
  const char* last_modified = "last-modified: ";
  const uint16_t last_modified_position =
      pbuf_memfind(headers, last_modified, strlen(last_modified), /*offset=*/0);
  const char* newline = "\n";
  const uint16_t newline_position =
      pbuf_memfind(headers, newline, strlen(newline), last_modified_position);
  memset(g_scum_bootloader_binary_last_modified, 0,
         sizeof(char) * SCUM_BINARY_LAST_MODIFIED_SIZE);
  g_scum_bootloader_binary_last_modified_size = pbuf_copy_partial(
      headers, g_scum_bootloader_binary_last_modified,
      /*len=*/newline_position - last_modified_position - strlen(last_modified),
      /*offset=*/last_modified_position + strlen(last_modified));
  return ERR_OK;
}

// Save the SCuM binary received within the HTTP body.
static err_t scum_bootloader_http_receive_binary_callback(
    void* arg, struct altcp_pcb* connection, struct pbuf* packet,
    const err_t error) {
  const uint16_t size =
      (g_scum_bootloader_binary_size + packet->tot_len > SCUM_BINARY_SIZE)
          ? (SCUM_BINARY_SIZE - g_scum_bootloader_binary_size)
          : packet->tot_len;
  g_scum_bootloader_binary_size += pbuf_copy_partial(
      packet, &g_scum_bootloader_binary[g_scum_bootloader_binary_size], size,
      /*offset=*/0);
  return ERR_OK;
}

// Return whether a new SCuM binary exists.
static bool scum_bootloader_has_new_binary() {
  // Copy the last modified time of the SCuM binary.
  char old_binary_last_modified[SCUM_BINARY_LAST_MODIFIED_SIZE];
  memset(old_binary_last_modified, 0,
         sizeof(char) * SCUM_BINARY_LAST_MODIFIED_SIZE);
  memcpy(old_binary_last_modified, g_scum_bootloader_binary_last_modified,
         g_scum_bootloader_binary_last_modified_size);

  // Check the last modified time of the SCuM binary.
  http_client_request_t request = {0};
  request.hostname = SCUM_BINARY_HOSTNAME;
  request.url = SCUM_BINARY_URL;
  request.headers_callback = scum_bootloader_http_receive_headers_callback;
  const int error =
      http_client_request_sync(cyw43_arch_async_context(), &request);
  if (error != 0) {
    printf("HTTP request failed with error %d.\n", error);
    return false;
  }
  if (g_scum_bootloader_binary_last_modified_size == 0) {
    return true;
  }
  return memcmp(g_scum_bootloader_binary_last_modified,
                old_binary_last_modified,
                g_scum_bootloader_binary_last_modified_size) != 0;
}

int main(int argc, char** argv) {
  // Initialize UART.
  stdio_uart_init_full(SCUM_UART_INSTANCE, SCUM_UART_BAUD_RATE, /*tx_pin=*/-1,
                       /*rx_pin=*/SCUM_UART_RX_PIN);
  gpio_disable_pulls(SCUM_UART_RX_PIN);

  // Initialize USB.
  stdio_usb_init();

  // Limit input and output to USB only.
  stdio_filter_driver(&stdio_usb);

  // Initialize GPIOs and the LED.
  scum_bootloader_gpio_init();
  scum_bootloader_led_init();

  // Initialize the UART buffer.
  queue_init(&g_scum_uart_buffer, sizeof(char), SCUM_UART_BUFFER_SIZE);

  // Initialize the WiFi.
  cyw43_arch_init();
  cyw43_arch_enable_sta_mode();

  // Connect to the WiFi network.
  if (cyw43_arch_wifi_connect_timeout_ms(
          WIFI_SSID, WIFI_PASSWORD, WIFI_ENCRYPTION, WIFI_CONNECT_TIMEOUT_MS)) {
    printf("Failed to connect to WiFi.\n");
    return EXIT_FAILURE;
  }
  printf("Connected to WiFi network %s.\n", WIFI_SSID);

  if (!add_repeating_timer_ms(SCUM_BINARY_LAST_MODIFIED_PERIOD_MS,
                              scum_binary_last_modified_timer_callback,
                              /*user_data=*/NULL,
                              &g_scum_bootloader_binary_last_modified_timer)) {
    printf("Failed to create the SCuM binary last modified time timer.\n");
    return EXIT_FAILURE;
  }

  g_scum_bootloader_state = STATE_IDLE;
  while (true) {
    switch (g_scum_bootloader_state) {
      case STATE_IDLE: {
        // Print out any UART from SCuM.
        if (!queue_is_empty(&g_scum_uart_buffer)) {
          char data = 0;
          while (queue_try_remove(&g_scum_uart_buffer, &data)) {
            printf("%c", data);
          }
        }
        break;
      }
      case STATE_CHECK_FOR_NEW_BINARY: {
        // Check if a new SCuM binary exists.
        if (scum_bootloader_has_new_binary()) {
          multicore_reset_core1();
          printf("Found new SCuM binary with last modified time of %s.\n",
                 g_scum_bootloader_binary_last_modified);
          g_scum_bootloader_state = STATE_DOWNLOAD_BINARY;
        } else {
          g_scum_bootloader_state = STATE_IDLE;
        }
        break;
      }
      case STATE_DOWNLOAD_BINARY: {
        printf("Downloading SCuM binary http://%s%s.\n", SCUM_BINARY_HOSTNAME,
               SCUM_BINARY_URL);
        memset(g_scum_bootloader_binary, 0, sizeof(uint8_t) * SCUM_BINARY_SIZE);
        g_scum_bootloader_binary_size = 0;

        // Send the HTTP request.
        http_client_request_t request = {0};
        request.hostname = SCUM_BINARY_HOSTNAME;
        request.url = SCUM_BINARY_URL;
        request.receive_callback = scum_bootloader_http_receive_binary_callback;
        const int error =
            http_client_request_sync(cyw43_arch_async_context(), &request);
        printf("Downloaded %u bytes for the SCuM binary.\n",
               g_scum_bootloader_binary_size);
        if (error != 0) {
          printf("HTTP request failed with error %d.\n", error);
        } else {
          g_scum_bootloader_state = STATE_HARD_RESET;
        }
        break;
      }
      case STATE_HARD_RESET: {
        printf("Executing a hard reset.\n");
        gpio_put(SCUM_CLOCK_PIN, false);
        gpio_put(SCUM_DATA_PIN, false);
        gpio_put(SCUM_ENABLE_PIN, false);

        // Execute a hard reset.
        gpio_set_dir(SCUM_HRESET_PIN, GPIO_OUT);
        gpio_put(SCUM_HRESET_PIN, false);
        sleep_us(SCUM_HARD_RESET_SLEEP_TIME_US);
        gpio_set_dir(SCUM_HRESET_PIN, GPIO_IN);
        sleep_us(SCUM_HARD_RESET_SLEEP_TIME_US);

        g_scum_bootloader_state = STATE_WRITE_BINARY;
        break;
      }
      case STATE_WRITE_BINARY: {
        printf("Writing the binary to SCuM.\n");
        gpio_put(PICO_LED_PIN, true);
        for (size_t i = 0; i < SCUM_BINARY_SIZE; ++i) {
          for (uint8_t j = 0; j < 8; ++j) {
            // Output the data.
            gpio_put(SCUM_DATA_PIN,
                     ((g_scum_bootloader_binary[i] >> j) & 0x1) == 0x1);
            // Toggle the enable pin after 32 bits.
            sleep_us(SCUM_TOGGLE_SLEEP_TIME_US);
            gpio_put(SCUM_ENABLE_PIN, ((i + 1) % 4 == 0) && (j == 7));
            // Toggle the clock pin.
            sleep_us(SCUM_TOGGLE_SLEEP_TIME_US);
            gpio_put(SCUM_CLOCK_PIN, true);
            sleep_us(SCUM_TOGGLE_SLEEP_TIME_US);
            gpio_put(SCUM_CLOCK_PIN, false);
            sleep_us(SCUM_TOGGLE_SLEEP_TIME_US);
          }
        }
        gpio_put(PICO_LED_PIN, false);
        g_scum_bootloader_state = STATE_CALIBRATION;
        break;
      }
      case STATE_CALIBRATION: {
        printf("Calibrating SCuM's oscillators.\n");
        if (!add_repeating_timer_ms(SCUM_CALIBRATION_CLOCK_PERIOD_MS,
                                    scum_calibration_timer_callback,
                                    /*user_data=*/NULL,
                                    &g_scum_calibration_timer)) {
          printf("Failed to create the calibration timer.\n");
          return EXIT_FAILURE;
        }
        while (g_scum_calibration_num_pulses < SCUM_CALIBRATION_NUM_PULSES) {
          // TODO(titan): Use a condition variable once it is implemented in the
          // pico_sync library.
          sleep_us(SCUM_CALIBRATION_CLOCK_ACTIVE_TIME_US);
        }
        g_scum_calibration_num_pulses = 0;
        g_scum_bootloader_state = STATE_UART_MONITOR;
        break;
      }
      case STATE_UART_MONITOR: {
        printf("SCuM bootloading complete.\n");
        // Launch the UART reader on core 1.
        multicore_launch_core1(scum_uart_reader);
        g_scum_bootloader_state = STATE_IDLE;
      }
      default: {
        break;
      }
    }
  }

  cyw43_arch_deinit();
  return EXIT_SUCCESS;
}
