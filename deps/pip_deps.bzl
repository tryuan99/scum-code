load("@python3_9//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_install")

def load_pip_dependencies():
    pip_install(
        name = "pip_deps",
        python_interpreter_target = interpreter,
        requirements = "//deps:pip_requirements.txt",
    )
