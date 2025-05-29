// This file contains common definitions for the Pico boards.

#ifndef PICO_DEFS_H_
#define PICO_DEFS_H_

// LED pin.
#ifdef PICO_W_ENABLED
#define PICO_LED_PIN CYW43_WL_GPIO_LED_PIN
#else  // !defined(PICO_W_ENABLED)
#define PICO_LED_PIN PICO_DEFAULT_LED_PIN
#endif  // PICO_W_ENABLED

#endif  // PICO_DEFS_H_
