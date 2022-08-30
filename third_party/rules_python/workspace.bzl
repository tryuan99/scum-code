load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_PYTHON_VERSION = "0.12.0"

def rules_python_workspace():
    http_archive(
        name = "rules_python",
        sha256 = "3c7480fe73c9712de4728d03bc4d4dd05c2b4626dfce8e5d9283197774469489",
        strip_prefix = "rules_python-0.12.0",
        url = "https://github.com/bazelbuild/rules_python/archive/0.12.0.zip",
    )
