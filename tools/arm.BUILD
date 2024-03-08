package(default_visibility = ["//visibility:public"])

filegroup(
    name = "compiler_support_files",
    srcs = glob([
        "arm-none-eabi/include/**",
        "arm-none-eabi/lib/**",
        "arm-none-eabi/11.3.1/**",
    ]),
)

filegroup(
    name = "compiler_files",
    srcs = [
        "bin/arm-none-eabi-ar",
        "bin/arm-none-eabi-as",
        "bin/arm-none-eabi-cpp",
        "bin/arm-none-eabi-gcc",
        "bin/arm-none-eabi-gcov",
        "bin/arm-none-eabi-ld",
        "bin/arm-none-eabi-nm",
        "bin/arm-none-eabi-objcopy",
        "bin/arm-none-eabi-objdump",
        "bin/arm-none-eabi-readelf",
        "bin/arm-none-eabi-strip",
        ":compiler_support_files",
    ],
)

filegroup(
    name = "ar_files",
    srcs = ["bin/arm-none-eabi-ar"],
)

filegroup(
    name = "as_files",
    srcs = ["bin/arm-none-eabi-as"],
)

filegroup(
    name = "linker_files",
    srcs = [
        "bin/arm-none-eabi-ld",
        ":compiler_files",
    ],
)

filegroup(
    name = "objcopy_files",
    srcs = ["bin/arm-none-eabi-objcopy"],
)

filegroup(
    name = "strip_files",
    srcs = ["bin/arm-none-eabi-strip"],
)

filegroup(
    name = "all_files",
    srcs = [
        ":ar_files",
        ":as_files",
        ":compiler_files",
        ":linker_files",
        ":objcopy_files",
        ":strip_files",
    ],
)

toolchain(
    name = "toolchain-firmware-scum-arm",
    target_compatible_with = [
        "@//tools/ti:ti-c674",
    ],
    toolchain = ":c674",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain(
    name = "c674",
    all_files = ":all_files",
    ar_files = ":ar_files",
    as_files = ":as_files",
    compiler_files = ":compiler_files",
    coverage_files = ":coverage_files",
    dwp_files = ":empty",
    linker_files = ":linker_files",
    objcopy_files = ":objcopy_files",
    strip_files = ":strip_files",
    supports_param_files = 0,
    toolchain_config = ":c674-config",
    toolchain_identifier = "arm",
)

cc_toolchain_config(
    name = "c674-config",
    assembler_flags = [
        "-mcpu=cortex-m0",
        "-mthumb",
    ],
    compiler_flags = [
        "-c",
        "-Wall",
        "-Werror",
        "-O0",
        "-nostdlib",
        "-nostartfiles",
        "-ffreestanding",
        "-mthumb",
        "-mcpu=cortex=m0",
        "-march=armv6s-m",
    ],
    linker_flags = [
        "-A=armv6-m",
        "-e Reset_Handler",
    ],
)

filegroup(
    name = "empty",
    srcs = [],
)
