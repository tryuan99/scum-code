load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "ping_pong_main",
    srcs = ["ping_pong_main.py"],
    deps = [
        requirement("absl-py"),
        requirement("pyserial"),
    ],
)
