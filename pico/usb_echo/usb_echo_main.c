#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "hardware/gpio.h"
#include "pico/error.h"
#include "pico/stdio.h"
#include "pico/stdio_usb.h"

// USB read timeout in microseconds.
#define USB_READ_TIMEOUT_US 1000

// Initialize the LED.
static inline void usb_echo_led_init() {
  gpio_init(PICO_DEFAULT_LED_PIN);
  gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
}

// Receive a byte over USB.
static inline bool usb_echo_receive_byte(uint8_t* rx_byte) {
  int read_byte = stdio_getchar_timeout_us(USB_READ_TIMEOUT_US);
  if (read_byte == PICO_ERROR_TIMEOUT) {
    return false;
  }
  *rx_byte = (uint8_t)(read_byte & 0xFF);
  return true;
}

int main(int argc, char** argv) {
  stdio_usb_init();
  usb_echo_led_init();

  uint8_t rx_byte = 0;
  while (true) {
    if (usb_echo_receive_byte(&rx_byte)) {
      gpio_put(PICO_DEFAULT_LED_PIN, true);
      printf("%c", rx_byte);
      gpio_put(PICO_DEFAULT_LED_PIN, false);
    }
  }

  return EXIT_SUCCESS;
}
