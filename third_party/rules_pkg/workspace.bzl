load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_PKG_VERSION = "0.9.1"

def rules_pkg_workspace():
    http_archive(
        name = "rules_pkg",
        sha256 = "360c23a88ceaf7f051abc99e2e6048cf7fe5d9af792690576554a88b2013612d",
        strip_prefix = "rules_pkg-{}".format(RULES_PKG_VERSION),
        url = "https://github.com/bazelbuild/rules_pkg/archive/refs/tags/{}.tar.gz".format(RULES_PKG_VERSION),
    )
