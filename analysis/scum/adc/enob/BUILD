load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "adc_data",
    srcs = ["adc_data.py"],
    deps = [requirement("numpy")],
)

py_binary(
    name = "adc_enob_main",
    srcs = ["adc_enob_main.py"],
    data = ["//analysis/scum/adc/enob/data:adc_enob_data"],
    deps = [
        ":adc_data",
        "//analysis/scum/adc:adc_config",
        "//utils/regression:linear_regression",
        "//utils/regression:parabolic_regression",
        requirement("absl-py"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("scipy"),
    ],
)

py_binary(
    name = "adc_enob_scope_cleanup_main",
    srcs = ["adc_enob_scope_cleanup_main.py"],
    deps = [
        requirement("absl-py"),
        requirement("numpy"),
        requirement("pandas"),
    ],
)

py_binary(
    name = "adc_enob_scope_main",
    srcs = ["adc_enob_scope_main.py"],
    data = ["//analysis/scum/adc/enob/data:adc_enob_data_with_scope"],
    deps = [
        ":adc_data",
        "//analysis/scum/adc:adc_config",
        requirement("absl-py"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pandas"),
    ],
)

py_binary(
    name = "adc_enob_scope_averaging_main",
    srcs = ["adc_enob_scope_averaging_main.py"],
    data = ["//analysis/scum/adc/enob/data:adc_enob_data_with_scope"],
    deps = [
        ":adc_data",
        "//analysis/scum/adc:adc_config",
        requirement("absl-py"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pandas"),
    ],
)

py_binary(
    name = "adc_enob_scope_vs_frequency_main",
    srcs = ["adc_enob_scope_vs_frequency_main.py"],
    deps = [
        requirement("absl-py"),
        requirement("matplotlib"),
        requirement("numpy"),
    ],
)
