from setuptools import Extension, setup

module = Extension(
    "symnmfmodule",
    sources=["symnmfmodule.c", "symnmf.c"],
    extra_compile_args=["-Wall", "-Wextra", "-Werror"],
    extra_link_args=["-lm"],
)

setup(
    name="symnmfmodule",
    version="1.0",
    description="Symmetric NMF C extension",
    ext_modules=[module],
)
