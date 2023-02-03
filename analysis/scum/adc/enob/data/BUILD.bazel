package(default_visibility = ["//visibility:public"])

filegroup(
    name = "adc_enob_data",
    srcs = glob(
        ["adc_data_*.csv"],
        exclude = ["adc_data_with_scope_*.csv"],
    ),
)

filegroup(
    name = "adc_enob_data_with_scope",
    srcs = glob(["adc_data_with_scope_*.csv"]) + glob(["scope_*.csv"]),
)
