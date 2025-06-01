// The SCuM bootloader bootloads SCuM using the 3-wire bus protocol. After
// bootloading, SCuM's oscillators are calibrated.

#ifndef PICO_SCUM_BOOTLOADER_H_
#define PICO_SCUM_BOOTLOADER_H_

#include <stdint.h>

// SCuM binary size in bytes.
// SCuM has a program memory size of 64 KiB.
#define SCUM_BINARY_SIZE (1 << 16)

// SCuM binary.
typedef struct {
  // SCuM binary data.
  uint8_t data[SCUM_BINARY_SIZE];
} scum_binary_t;

// SCuM bootloading complete callback function type.
typedef void (*scum_bootloader_complete_function_t)();

// Initialize the SCuM bootloader.
void scum_bootloader_init();

// Bootload SCuM with the given binary. The complete callback function will be
// called after bootloading is complete.
void scum_bootloader_run(const scum_binary_t* binary,
                         scum_bootloader_complete_function_t complete_callback);

// Run the SCuM bootloader loop. This function should be called from within a
// while loop.
void scum_bootloader_loop();

#endif  // PICO_SCUM_BOOTLOADER_H_
