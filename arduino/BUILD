load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "ping_pong_main",
    srcs = ["ping_pong_main.py"],
    deps = [
        "//utils/serial:serial_interface",
        requirement("absl-py"),
    ],
)
