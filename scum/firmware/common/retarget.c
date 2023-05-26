#include <stdbool.h>
#include <stdio.h>

#define APB_UART_BASE 0x51000000

struct __FILE {
  unsigned char* ptr;
};

FILE __stdout = {(unsigned char*)APB_UART_BASE};
FILE __stdin = {(unsigned char*)APB_UART_BASE};

int uart_out(int ch) {
  unsigned char* UARTPtr = (unsigned char*)APB_UART_BASE;
  *UARTPtr = (char)ch;
  return ch;
}

int uart_in() {
  unsigned char* UARTPtr = (unsigned char*)APB_UART_BASE;
  char ch = *UARTPtr;
  uart_out(ch);
  return (int)ch;
}

int fputc(int ch, FILE* f) { return uart_out(ch); }

int fgetc(FILE* f) { return uart_in(); }

// int ferror(FILE *f) {
//     return 0;
// }

void _ttywrch(int ch) { fputc(ch, &__stdout); }

void _sys_exit(void) {
  printf("\nTEST DONE\n");
  while (true)
    ;
}
