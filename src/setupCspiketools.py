from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("Cspiketools", ["Cspiketools.pyx"],
        include_dirs=[numpy.get_include()]),
]
setup(
    name="Cspiketools",
    ext_modules=cythonize(extensions),
)