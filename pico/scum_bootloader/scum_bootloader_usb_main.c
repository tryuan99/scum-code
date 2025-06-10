#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "pico/defs.h"
#include "pico/error.h"
#include "pico/scum/bootloader.h"
#include "pico/scum/uart.h"
#include "pico/stdio.h"
#include "pico/stdio_usb.h"

// USB read timeout in microseconds.
#define SCUM_USB_READ_TIMEOUT_US 1000

// SCuM USB bootloading state enumeration.
typedef enum {
  STATE_INVALID = -1,
  STATE_INIT,
  STATE_IDLE,
  STATE_RECEIVE_BINARY,
  STATE_BOOTLOAD,
  STATE_DONE,
} scum_bootloader_usb_state_e;

// SCuM USB bootloading state.
static scum_bootloader_usb_state_e g_scum_bootloader_usb_state = STATE_INIT;

// SCuM binary.
static scum_binary_t g_scum_bootloader_usb_binary;

// SCuM binary size in bytes.
static size_t g_scum_bootloader_usb_binary_size = 0;

// Receive a byte of the binary over USB.
static inline bool scum_bootloader_usb_receive_byte(uint8_t* data) {
  const int received_byte = stdio_getchar_timeout_us(SCUM_USB_READ_TIMEOUT_US);
  if (received_byte == PICO_ERROR_TIMEOUT) {
    return false;
  }
  *data = (uint8_t)(received_byte & 0xFF);
  return true;
}

// SCuM bootloading complete callback function.
static void scum_bootloader_usb_complete_callback(void) {
  g_scum_bootloader_usb_state = STATE_DONE;
}

int main(int argc, char** argv) {
  // Initialize USB.
  stdio_usb_init();

  // Limit input and output to USB only.
  stdio_filter_driver(&stdio_usb);

  // Initialize SCuM's UART.
  scum_uart_init();

  // Initialize the SCuM bootloader..
  scum_bootloader_init();

  g_scum_bootloader_usb_state = STATE_IDLE;
  while (true) {
    switch (g_scum_bootloader_usb_state) {
      case STATE_IDLE: {
        scum_uart_print();

        // Check if a new SCuM binary is being received.
        if (scum_bootloader_usb_receive_byte(
                &g_scum_bootloader_usb_binary
                     .data[g_scum_bootloader_usb_binary_size])) {
          ++g_scum_bootloader_usb_binary_size;
          scum_uart_stop();
          g_scum_bootloader_usb_state = STATE_RECEIVE_BINARY;
        }
      }
      case STATE_RECEIVE_BINARY: {
        if (g_scum_bootloader_usb_binary_size == SCUM_BINARY_SIZE) {
          scum_bootloader_run(&g_scum_bootloader_usb_binary,
                              scum_bootloader_usb_complete_callback);
          g_scum_bootloader_usb_binary_size = 0;
          g_scum_bootloader_usb_state = STATE_BOOTLOAD;
        } else if (scum_bootloader_usb_receive_byte(
                       &g_scum_bootloader_usb_binary
                            .data[g_scum_bootloader_usb_binary_size])) {
          ++g_scum_bootloader_usb_binary_size;
        }
        break;
      }
      case STATE_BOOTLOAD: {
        scum_bootloader_loop();
        break;
      }
      case STATE_DONE: {
        scum_uart_start();
        g_scum_bootloader_usb_state = STATE_IDLE;
        break;
      }
      default: {
        break;
      }
    }
  }

  return EXIT_SUCCESS;
}
