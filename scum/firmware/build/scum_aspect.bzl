"""The SCuM aspect is used to generate additional artifacts, such as a binary
file or an Intel hex file.
"""

def _scum_bin_aspect_impl(target, ctx):
    binary = target[DefaultInfo].files_to_run.executable
    bin_output = ctx.actions.declare_file(binary.basename + ".bin")
    ctx.actions.run(
        outputs = [bin_output],
        inputs = [binary],
        tools = [ctx.executable._objcopy],
        executable = ctx.executable._objcopy,
        arguments = [
            "-Obinary",
            binary.path,
            bin_output.path,
        ],
    )
    return [
        OutputGroupInfo(
            scum_bin_files = depset([bin_output]),
        ),
    ]

scum_bin_aspect = aspect(
    implementation = _scum_bin_aspect_impl,
    attrs = {
        "_objcopy": attr.label(default = "@arm_none_eabi//:objcopy", executable = True, cfg = "exec"),
    },
)

def _scum_hex_aspect_impl(target, ctx):
    binary = target[DefaultInfo].files_to_run.executable
    hex_output = ctx.actions.declare_file(binary.basename + ".hex")
    ctx.actions.run(
        outputs = [hex_output],
        inputs = [binary],
        tools = [ctx.executable._objcopy],
        executable = ctx.executable._objcopy,
        arguments = [
            "-Oihex",
            binary.path,
            hex_output.path,
        ],
    )
    return [
        OutputGroupInfo(
            scum_hex_files = depset([hex_output]),
        ),
    ]

scum_hex_aspect = aspect(
    implementation = _scum_hex_aspect_impl,
    attrs = {
        "_objcopy": attr.label(default = "@arm_none_eabi//:objcopy", executable = True, cfg = "exec"),
    },
)
