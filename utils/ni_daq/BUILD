load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "voltage_logger",
    srcs = ["voltage_logger.py"],
    deps = [requirement("absl-py")],
)

py_binary(
    name = "voltage_logger_main",
    srcs = ["voltage_logger_main.py"],
    deps = [
        ":voltage_logger",
        requirement("absl-py"),
    ],
)
