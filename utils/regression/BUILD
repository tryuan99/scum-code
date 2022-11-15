load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "exponential_regression",
    srcs = ["exponential_regression.py"],
    deps = [
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_library(
    name = "linear_regression",
    srcs = ["linear_regression.py"],
    deps = [requirement("numpy")],
)

py_library(
    name = "parabolic_regression",
    srcs = ["parabolic_regression.py"],
    deps = [requirement("numpy")],
)
