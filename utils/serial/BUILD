load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "serial_interface",
    srcs = ["serial_interface.py"],
    deps = [
        requirement("absl-py"),
        requirement("pyserial"),
    ],
)

py_library(
    name = "serial_logger",
    srcs = ["serial_logger.py"],
    deps = [
        ":serial_interface",
        requirement("absl-py"),
    ],
)

py_binary(
    name = "serial_logger_main",
    srcs = ["serial_logger_main.py"],
    deps = [
        ":serial_logger",
        requirement("absl-py"),
    ],
)

py_library(
    name = "serial_monitor",
    srcs = ["serial_monitor.py"],
    deps = [
        ":serial_interface",
        requirement("absl-py"),
    ],
)

py_binary(
    name = "serial_monitor_main",
    srcs = ["serial_monitor_main.py"],
    deps = [
        ":serial_monitor",
        requirement("absl-py"),
    ],
)
