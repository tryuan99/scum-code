load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "nrf_bootloader",
    srcs = ["nrf_bootloader.py"],
    deps = [
        "//utils:serial_interface",
        requirement("absl-py"),
    ],
)

py_binary(
    name = "nrf_bootloader_main",
    srcs = ["nrf_bootloader_main.py"],
    deps = [
        ":nrf_bootloader",
        requirement("absl-py"),
    ],
)
