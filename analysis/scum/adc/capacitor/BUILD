load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "adc_capacitor_main",
    srcs = ["adc_capacitor_main.py"],
    data = ["//analysis/scum/adc/capacitor/data:adc_scope_capacitor_data"],
    deps = [
        "//utils/regression:linear_regression",
        requirement("absl-py"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pandas"),
    ],
)

py_binary(
    name = "adc_capacitor_scope_cleanup_main",
    srcs = ["adc_capacitor_scope_cleanup_main.py"],
    deps = [
        requirement("absl-py"),
        requirement("numpy"),
        requirement("pandas"),
    ],
)
