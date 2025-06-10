#include "pico/scum/uart.h"

#include <stdio.h>

#include "hardware/gpio.h"
#include "hardware/uart.h"
#include "pico/multicore.h"
#include "pico/stdio_uart.h"
#include "pico/util/queue.h"

// UART configuration for SCuM.
#define SCUM_UART_INSTANCE uart0
#define SCUM_UART_RX_PIN 1
#define SCUM_UART_BAUD_RATE 19200
#define SCUM_UART_BUFFER_SIZE 512

// UART buffer.
static queue_t g_scum_uart_buffer;

// SCuM UART reader.
static void scum_uart_reader(void) {
  while (true) {
    if (uart_is_readable(SCUM_UART_INSTANCE)) {
      char data = uart_getc(SCUM_UART_INSTANCE);
      queue_try_add(&g_scum_uart_buffer, &data);
    }
  }
}

void scum_uart_init(void) {
  // Initialize the UART pins.
  stdio_uart_init_full(SCUM_UART_INSTANCE, SCUM_UART_BAUD_RATE, /*tx_pin=*/-1,
                       /*rx_pin=*/SCUM_UART_RX_PIN);
  gpio_disable_pulls(SCUM_UART_RX_PIN);

  // Initialize the UART buffer.
  queue_init(&g_scum_uart_buffer, sizeof(char), SCUM_UART_BUFFER_SIZE);
}

void scum_uart_start(void) { multicore_launch_core1(scum_uart_reader); }

void scum_uart_stop(void) { multicore_reset_core1(); }

void scum_uart_print(void) {
  if (!queue_is_empty(&g_scum_uart_buffer)) {
    char data = 0;
    while (queue_try_remove(&g_scum_uart_buffer, &data)) {
      printf("%c", data);
    }
  }
}
