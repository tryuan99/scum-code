// This file contains various utilities for SCuM.

#ifndef __UTILS_H_
#define __UTILS_H_

#include <stdint.h>
#include <stdio.h>

// Loop for the given number of cycles.
static inline void utils_busy_wait_cycles(uint32_t cycles) {
  while (cycles--) {
    __asm__ volatile("" :::);
  }
}

#endif  // __UTILS_H_
