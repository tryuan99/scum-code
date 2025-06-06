#ifndef __HELPERS_H
#define __HELPERS_H

#include <stdint.h>
#include <stdio.h>

/** @brief Loop for a number of cycles.
 *
 *@param[in] cycles Number of cycles to loop for.
 **/
static inline void busy_wait_cycles(uint32_t cycles) {
  while (cycles--) {
    __asm__ volatile("" :::);
  }
}

#endif
