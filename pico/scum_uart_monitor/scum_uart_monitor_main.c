#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "hardware/gpio.h"
#include "pico/error.h"
#include "pico/stdio.h"
#include "pico/stdio_uart.h"
#include "pico/stdio_usb.h"

// UART configuration for SCuM.
#define SCUM_UART_INSTANCE uart0
#define SCUM_UART_RX_PIN 1
#define SCUM_UART_BAUD_RATE 19200

// UART read timeout in microseconds.
#define SCUM_UART_READ_TIMEOUT_US 100

// Initialize the LED.
static inline void scum_uart_led_init() {
  gpio_init(PICO_DEFAULT_LED_PIN);
  gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
}

// Receive a byte over UART from SCuM.
static inline bool scum_uart_receive_byte(uint8_t* data) {
  int received_byte = stdio_getchar_timeout_us(SCUM_UART_READ_TIMEOUT_US);
  if (received_byte == PICO_ERROR_TIMEOUT) {
    return false;
  }
  *data = (uint8_t)(received_byte & 0xFF);
  return true;
}

int main(int argc, char** argv) {
  // Initialize UART.
  stdio_uart_init_full(SCUM_UART_INSTANCE, SCUM_UART_BAUD_RATE, /*tx_pin=*/-1,
                       /*rx_pin=*/SCUM_UART_RX_PIN);
  gpio_disable_pulls(SCUM_UART_RX_PIN);

  // Initialize USB.
  stdio_usb_init();

  // Initialize the LED.
  scum_uart_led_init();

  uint8_t data = 0;
  while (true) {
    if (scum_uart_receive_byte(&data)) {
      gpio_put(PICO_DEFAULT_LED_PIN, true);
      printf("%c", data);
      gpio_put(PICO_DEFAULT_LED_PIN, false);
    }
  }

  return EXIT_SUCCESS;
}
