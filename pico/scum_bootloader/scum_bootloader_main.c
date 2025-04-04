#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "hardware/gpio.h"
#include "pico/error.h"
#include "pico/stdio.h"
#include "pico/stdio_usb.h"
#include "pico/time.h"

// SCuM binary size in bytes.
// SCuM has a program memory size of 64 KiB.
#define SCUM_BINARY_SIZE (1 << 16)

// GPIO pins to SCuM.
#define SCUM_CLOCK_PIN 2
#define SCUM_DATA_PIN 3
#define SCUM_ENABLE_PIN 4
#define SCUM_HRESET_PIN 5

// Calibration clock period in milliseconds.
#define SCUM_CALIBRATION_CLOCK_PERIOD_MS 100

// Calibration number of pulses.
#define SCUM_CALIBRATION_NUM_PULSES 30

// USB read timeout in microseconds.
#define USB_READ_TIMEOUT_US 1000

// Hard reset sleep time in milliseconds.
#define HARD_RESET_SLEEP_TIME_MS 14

// Toggle sleep time in microseconds.
#define TOGGLE_SLEEP_TIME_US 1

// SCuM bootloading state enumeration.
typedef enum {
  STATE_INVALID = -1,
  STATE_INIT,
  STATE_RECEIVING_BINARY,
  STATE_HARD_RESET,
  STATE_WRITING_BINARY,
  STATE_CALIBRATING,
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
static repeating_timer_t g_scum_bootloader_calibration_timer;

// Calibration number of pulses.
static uint32_t g_scum_bootloader_calibration_num_pulses = 0;

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
}

// Initialize the LED.
static inline void scum_bootloader_led_init() {
  gpio_init(PICO_DEFAULT_LED_PIN);
  gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
}

// Receive a byte of the binary over USB.
static inline bool scum_bootloader_receive_byte(uint8_t* rx_byte) {
  int read_byte = stdio_getchar_timeout_us(USB_READ_TIMEOUT_US);
  if (read_byte == PICO_ERROR_TIMEOUT) {
    return false;
  }
  *rx_byte = (uint8_t)(read_byte & 0xFF);
  return true;
}

// Calibration timer callback.
static bool scum_bootloader_calibration_timer_callback(
    repeating_timer_t* timer) {
  bool clock_state = gpio_get(SCUM_CLOCK_PIN);
  gpio_put(SCUM_CLOCK_PIN, !clock_state);
  if (clock_state) {
    ++g_scum_bootloader_calibration_num_pulses;
  }
}

int main(int argc, char** argv) {
  stdio_usb_init();
  scum_bootloader_gpio_init();
  scum_bootloader_led_init();
  g_scum_bootloader_state = STATE_RECEIVING_BINARY;

  while (true) {
    switch (g_scum_bootloader_state) {
      case STATE_RECEIVING_BINARY: {
        if (scum_bootloader_receive_byte(
                &g_scum_bootloader_binary[g_scum_bootloader_binary_size])) {
          ++g_scum_bootloader_binary_size;
          if (g_scum_bootloader_binary_size == SCUM_BINARY_SIZE) {
            g_scum_bootloader_binary_size = 0;
            g_scum_bootloader_state = STATE_HARD_RESET;
          }
        }
        break;
      }
      case STATE_HARD_RESET: {
        printf(RESPONSE_OK);
        gpio_put(SCUM_CLOCK_PIN, false);
        gpio_put(SCUM_DATA_PIN, false);
        gpio_put(SCUM_ENABLE_PIN, false);
        // Execute a hard reset.
        gpio_put(SCUM_HRESET_PIN, false);
        gpio_set_dir(SCUM_HRESET_PIN, GPIO_OUT);
        sleep_ms(HARD_RESET_SLEEP_TIME_MS);
        gpio_set_dir(SCUM_HRESET_PIN, GPIO_IN);
        sleep_ms(HARD_RESET_SLEEP_TIME_MS);
        g_scum_bootloader_state = STATE_WRITING_BINARY;
        break;
      }
      case STATE_WRITING_BINARY: {
        gpio_put(PICO_DEFAULT_LED_PIN, true);
        for (size_t i = 0; i < SCUM_BINARY_SIZE; ++i) {
          for (uint8_t j = 0; j < 8; ++j) {
            // Output the data.
            gpio_put(SCUM_DATA_PIN,
                     ((g_scum_bootloader_binary[i] >> j) & 0x1) == 0x1);
            // Toggle the enable pin.
            sleep_us(TOGGLE_SLEEP_TIME_US);
            gpio_put(SCUM_ENABLE_PIN, ((i + 1) % 4 == 0) && (j == 7));
            // Toggle the clock pin.
            sleep_us(TOGGLE_SLEEP_TIME_US);
            gpio_put(SCUM_CLOCK_PIN, true);
            sleep_us(TOGGLE_SLEEP_TIME_US);
            gpio_put(SCUM_CLOCK_PIN, false);
            sleep_us(TOGGLE_SLEEP_TIME_US);
          }
        }
        printf(RESPONSE_OK);
        gpio_put(PICO_DEFAULT_LED_PIN, false);
        g_scum_bootloader_state = STATE_CALIBRATING;
        break;
      }
      case STATE_CALIBRATING: {
        if (!add_repeating_timer_ms(SCUM_CALIBRATION_CLOCK_PERIOD_MS / 2,
                                    scum_bootloader_calibration_timer_callback,
                                    /*user_data=*/NULL,
                                    &g_scum_bootloader_calibration_timer)) {
          printf("Failed to create the calibration timer.\n");
          return EXIT_FAILURE;
        }
        while (g_scum_bootloader_calibration_num_pulses <
               SCUM_CALIBRATION_NUM_PULSES) {}
        if (!cancel_repeating_timer(&g_scum_bootloader_calibration_timer)) {
          printf("Failed to cancel the calibration timer.\n");
        }
        printf(RESPONSE_OK);
        g_scum_bootloader_calibration_num_pulses = 0;
        g_scum_bootloader_state = STATE_RECEIVING_BINARY;
        break;
      }
      default: {
        break;
      }
    }
  }

  return EXIT_SUCCESS;
}
