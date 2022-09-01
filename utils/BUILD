load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "serial_interface",
    srcs = ["serial_interface.py"],
    deps = [
        requirement("absl-py"),
        requirement("pyserial"),
    ],
)
