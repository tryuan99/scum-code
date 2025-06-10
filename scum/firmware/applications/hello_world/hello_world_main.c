#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "scum/firmware/common/utils.h"

// Number of for loop cycles between messages.
// 700000 for loop cycles roughly correspond to 1 second.
#define NUM_CYCLES_BETWEEN_PRINT 1000000

int main(int argc, char** argv) {
  uint32_t i = 0;

  while (true) {
    printf("Hello World! %lu\n", ++i);
    utils_busy_wait_cycles(NUM_CYCLES_BETWEEN_PRINT);
  }
}
