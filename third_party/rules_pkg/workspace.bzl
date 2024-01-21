load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_PKG_VERSION = "0.10.0"

def rules_pkg_workspace():
    http_archive(
        name = "rules_pkg",
        sha256 = "39d9b69b19cc5435d2650d23ce732f1c220ab0627dfd99782ddd6b3d82fe4cd4",
        strip_prefix = "rules_pkg-{}".format(RULES_PKG_VERSION),
        url = "https://github.com/bazelbuild/rules_pkg/archive/refs/tags/{}.tar.gz".format(RULES_PKG_VERSION),
    )
