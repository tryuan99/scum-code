#include "pico/scum/bootloader.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "hardware/gpio.h"
#include "pico/defs.h"
#include "pico/time.h"

// 3WB pins to SCuM.
#define SCUM_CLOCK_PIN 2
#define SCUM_DATA_PIN 3
#define SCUM_ENABLE_PIN 4
#define SCUM_HRESET_PIN 5

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
  STATE_HARD_RESET,
  STATE_WRITE_BINARY,
  STATE_CALIBRATION,
} scum_bootloader_state_e;

// SCuM bootloading state.
static scum_bootloader_state_e g_scum_bootloader_state = STATE_INIT;

// Pointer to the SCuM binary.
static const scum_binary_t* g_scum_binary = NULL;

// Calibration timer.
static repeating_timer_t g_scum_calibration_timer;

// Calibration number of pulses.
static uint32_t g_scum_calibration_num_pulses = 0;

// SCuM bootloading complete callback function pointer.
scum_bootloader_complete_function_t g_scum_bootloader_complete_callback;

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

void scum_bootloader_init() {
  // Initialize the GPIOs and the LED.
  scum_bootloader_gpio_init();
  scum_bootloader_led_init();
}

void scum_bootloader_run(
    const scum_binary_t* binary,
    scum_bootloader_complete_function_t complete_callback) {
  if (binary != NULL) {
    g_scum_binary = binary;
    g_scum_bootloader_complete_callback = complete_callback;
    g_scum_bootloader_state = STATE_HARD_RESET;
  }
}

void scum_bootloader_loop() {
  if (g_scum_binary == NULL) {
    g_scum_bootloader_state = STATE_IDLE;
  }

  switch (g_scum_bootloader_state) {
    case STATE_IDLE: {
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
          gpio_put(SCUM_DATA_PIN, ((g_scum_binary->data[i] >> j) & 0x1) == 0x1);
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
      if (!add_repeating_timer_ms(
              SCUM_CALIBRATION_CLOCK_PERIOD_MS, scum_calibration_timer_callback,
              /*user_data=*/NULL, &g_scum_calibration_timer)) {
        printf("Failed to create the calibration timer.\n");
        return;
      }
      while (g_scum_calibration_num_pulses < SCUM_CALIBRATION_NUM_PULSES) {
        // TODO(titan): Use a condition variable once it is implemented in the
        // pico_sync library.
        sleep_us(SCUM_CALIBRATION_CLOCK_ACTIVE_TIME_US);
      }
      g_scum_calibration_num_pulses = 0;

      if (g_scum_bootloader_complete_callback != NULL) {
        g_scum_bootloader_complete_callback();
      }
      g_scum_bootloader_state = STATE_IDLE;
      printf("SCuM bootloading complete.\n");
      break;
    }
    default: {
      break;
    }
  }
}
