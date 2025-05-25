#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "hardware/gpio.h"
#include "hardware/uart.h"
#include "pico/defs.h"
#include "pico/error.h"
#include "pico/multicore.h"
#include "pico/stdio.h"
#include "pico/stdio_uart.h"
#include "pico/stdio_usb.h"
#include "pico/time.h"
#include "pico/util/queue.h"

// SCuM binary size in bytes.
// SCuM has a program memory size of 64 KiB.
#define SCUM_BINARY_SIZE (1 << 16)

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

// USB read timeout in microseconds.
#define SCUM_USB_READ_TIMEOUT_US 1000

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
  STATE_RECEIVE_BINARY,
  STATE_HARD_RESET,
  STATE_WRITE_BINARY,
  STATE_CALIBRATION,
  STATE_UART_MONITOR,
} scum_bootloader_state_e;

// OK response.
static const char* RESPONSE_OK = "OK\n";

// SCuM bootloading state.
static scum_bootloader_state_e g_scum_bootloader_state = STATE_INIT;

// SCuM binary.
static uint8_t g_scum_bootloader_binary[SCUM_BINARY_SIZE];

// Number of bytes received.
static size_t g_scum_bootloader_binary_size = 0;

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

// Receive a byte of the binary over USB.
static inline bool scum_bootloader_receive_byte(uint8_t* data) {
  int received_byte = stdio_getchar_timeout_us(SCUM_USB_READ_TIMEOUT_US);
  if (received_byte == PICO_ERROR_TIMEOUT) {
    return false;
  }
  *data = (uint8_t)(received_byte & 0xFF);
  return true;
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

        // Check if a new SCuM binary is being received.
        if (scum_bootloader_receive_byte(
                &g_scum_bootloader_binary[g_scum_bootloader_binary_size])) {
          ++g_scum_bootloader_binary_size;
          multicore_reset_core1();
          g_scum_bootloader_state = STATE_RECEIVE_BINARY;
        }
      }
      case STATE_RECEIVE_BINARY: {
        if (g_scum_bootloader_binary_size == SCUM_BINARY_SIZE) {
          g_scum_bootloader_binary_size = 0;
          g_scum_bootloader_state = STATE_HARD_RESET;
        } else if (scum_bootloader_receive_byte(
                       &g_scum_bootloader_binary
                           [g_scum_bootloader_binary_size])) {
          ++g_scum_bootloader_binary_size;
        }
        break;
      }
      case STATE_HARD_RESET: {
        printf(RESPONSE_OK);
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
        printf(RESPONSE_OK);
        gpio_put(PICO_LED_PIN, false);
        g_scum_bootloader_state = STATE_CALIBRATION;
        break;
      }
      case STATE_CALIBRATION: {
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
        printf(RESPONSE_OK);
        g_scum_calibration_num_pulses = 0;
        g_scum_bootloader_state = STATE_UART_MONITOR;
        break;
      }
      case STATE_UART_MONITOR: {
        // Launch the UART reader on core 1.
        multicore_launch_core1(scum_uart_reader);
        g_scum_bootloader_state = STATE_IDLE;
      }
      default: {
        break;
      }
    }
  }

  return EXIT_SUCCESS;
}
