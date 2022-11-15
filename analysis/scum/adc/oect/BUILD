load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "adc_oect_main",
    srcs = ["adc_oect_main.py"],
    data = ["//analysis/scum/adc/oect/data:adc_oect_data"],
    deps = [
        "//analysis/scum/adc:adc_config",
        requirement("absl-py"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("regex"),
    ],
)
