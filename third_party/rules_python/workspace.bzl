load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_PYTHON_VERSION = "0.28.0"

def rules_python_workspace():
    http_archive(
        name = "rules_python",
        sha256 = "d70cd72a7a4880f0000a6346253414825c19cdd40a28289bdf67b8e6480edff8",
        strip_prefix = "rules_python-{}".format(RULES_PYTHON_VERSION),
        url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/{}.tar.gz".format(RULES_PYTHON_VERSION),
    )
