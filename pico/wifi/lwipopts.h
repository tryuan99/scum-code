// This file defines the options for lwIP.
// See https://www.nongnu.org/lwip/2_1_x/group__lwip__opts.html for more
// details.

#ifndef PICO_WIFI_LWIPOPTS_H_
#define PICO_WIFI_LWIPOPTS_H_

#include "pico/wifi/lwip_config.h"

// TCP WND must be at least 16 KB to match the TLS record size.
#undef TCP_WND
#define TCP_WND 16384

#define LWIP_ALTCP 1

#endif  // PICO_WIFI_LWIPOPTS_H_
