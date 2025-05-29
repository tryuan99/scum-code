// This file contains the environment variables for WiFi.

#ifndef PICO_WIFI_ENV_H_
#define PICO_WIFI_ENV_H_

#include "pico/cyw43_arch.h"

// WiFi SSID.
#define WIFI_SSID "WIFI SSID"

// WiFi password.
#define WIFI_PASSWORD "WIFI PASSWORD"

// WiFi encryption.
#define WIFI_ENCRYPTION CYW43_AUTH_WPA3_WPA2_AES_PSK

#endif  // PICO_WIFI_ENV_H_
