// The SCuM UART passes SCuM's UART output to the Pico's USB output.
// SCuM's UART reader is executed on core 1.

#ifndef PICO_SCUM_UART_H_
#define PICO_SCUM_UART_H_

// Initialize SCuM's UART output.
void scum_uart_init();

// Start SCuM's UART output.
void scum_uart_start();

// Stop SCuM's UART output.
void scum_uart_stop();

// Print SCuM's UART output. This function should be called from within a while
// loop.
void scum_uart_print();

#endif  // PICO_SCUM_UART_H_
