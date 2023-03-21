load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_PKG_VERSION = "0.8.1"

def rules_pkg_workspace():
    http_archive(
        name = "rules_pkg",
        sha256 = "99d56f7cba0854dd1db96cf245fd52157cef58808c8015e96994518d28e3c7ab",
        strip_prefix = "rules_pkg-{}".format(RULES_PKG_VERSION),
        url = "https://github.com/bazelbuild/rules_pkg/archive/refs/tags/{}.tar.gz".format(RULES_PKG_VERSION),
    )
