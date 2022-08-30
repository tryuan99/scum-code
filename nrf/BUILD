load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "bootload",
    srcs = ["bootload.py"],
    deps = [
        requirement("absl-py"),
        requirement("pyserial"),
    ],
)
