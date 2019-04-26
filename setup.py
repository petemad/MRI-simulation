from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('RD.pyx'))
setup(ext_modules = cythonize('rotation.pyx'))