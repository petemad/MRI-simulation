from distutils.core import setup
from Cython.Build import cythonize

#setup(ext_modules = cythonize('RD.pyx'))
#setup(ext_modules = cythonize('rotation.pyx'))
#setup(ext_modules = cythonize('se.pyx'))
#setup(ext_modules = cythonize('preparation.pyx'))
setup(ext_modules = cythonize('reconstruction.pyx'))
