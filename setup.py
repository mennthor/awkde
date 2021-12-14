"""
Note: `pip install --user pybind11` mandatory. Did not really test if the
instal_requires and import workaround below works. If not, just install it
manually first.

Credits:
- https://pybind11.readthedocs.io/en/latest/compiling.html
- https://github.com/pybind/python_example
"""

from setuptools import setup, find_packages

# pybind11.readthedocs.io/en/stable/compiling.html#classic-setup-requires
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    # If pybind11 is not installed, these are used first and in a second pass
    # done automatically the pybind11 is then installed as the requirement
    from setuptools import Extension as Pybind11Extension
    build_ext = None


__version__ = "0.1"

ext_modules = [
    Pybind11Extension(
        # Module name (toplevel same as python, then same as in C++ file)
        name="awkde.backend",
        sources=["cpp/backend.cpp"],
        extra_compile_args=["-O3"],  # "-ggdb" for debugging stuff
        ),
]

setup(
    name="awkde",
    version="0.1",
    description="Common scripts for E@H post-processing",
    author="Thorben Menne",
    author_email="thorben.menne@aei.mpg.de",
    url="https://gitlab.aei.uni-hannover.de/thmenn/postproc",
    packages=find_packages(),
    setup_requires=[
        "pybind11",  # For this install script
    ],
    install_requires=["numpy", "scipy", "scikit-learn", "pybind11", "future"],
    ext_modules=ext_modules,  # Compiled external modules
    cmdclass={
        "build_ext": build_ext  # Finds highest supported C++ version
    },
)
