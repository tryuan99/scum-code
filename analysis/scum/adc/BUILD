load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "adc_config",
    srcs = ["adc_config.py"],
)

py_binary(
    name = "adc_characterization_main",
    srcs = ["adc_characterization_main.py"],
    data = ["//analysis/scum/adc/data:adc_characterization_data"],
    deps = [
        ":adc_config",
        "//utils/regression:linear_regression",
        requirement("absl-py"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pandas"),
    ],
)
