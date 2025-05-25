#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "hardware/clocks.h"
#include "hardware/vreg.h"
#include "pico/cyw43_arch.h"
#include "pico/stdlib.h"

// Print the scan result.
static int scan_result(void* env, const cyw43_ev_scan_result_t* result) {
  if (result != NULL) {
    printf(
        "SSID: %-32s, RSSI: %4d, channel: %3d, MAC: "
        "%02x:%02x:%02x:%02x:%02x:%02x, security: %u\n",
        result->ssid, result->rssi, result->channel, result->bssid[0],
        result->bssid[1], result->bssid[2], result->bssid[3], result->bssid[4],
        result->bssid[5], result->auth_mode);
  }
  return 0;
}

int main(int argc, char** argv) {
  stdio_init_all();
  cyw43_arch_init();
  cyw43_arch_enable_sta_mode();

  absolute_time_t scan_time = nil_time;
  bool scan_in_progress = false;
  while (true) {
    if (absolute_time_diff_us(get_absolute_time(), scan_time) < 0) {
      if (!scan_in_progress) {
        cyw43_wifi_scan_options_t scan_options = {0};
        int error =
            cyw43_wifi_scan(&cyw43_state, &scan_options, NULL, scan_result);
        if (error == 0) {
          printf("Performing WiFi scan.\n");
          scan_in_progress = true;
        } else {
          printf("Failed to start WiFi scan: %d.\n", error);
          // Wait 10 seconds and scan again.
          scan_time = make_timeout_time_ms(10000);
        }
      } else if (!cyw43_wifi_scan_active(&cyw43_state)) {
        // Wait 10 seconds and scan again.
        scan_time = make_timeout_time_ms(10000);
        scan_in_progress = false;
      }
    }
  }

  cyw43_arch_deinit();
  return EXIT_SUCCESS;
}
