from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#setup(ext_modules = cythonize('RD.pyx'))
#setup(ext_modules = cythonize('rotation.pyx'))
#setup(ext_modules = cythonize('se.pyx'))
#setup(ext_modules = cythonize('preparation.pyx'))
#setup(ext_modules = cythonize('reconstruction.pyx'))
setup(ext_modules = cythonize('appc.pyx'))
#ext_modules=[
#    Extension("appc",
#              sources=["appc.pyx"],
#              libraries=["m"] # Unix-like specific
#    )
#]

#setup(
#  name = "appc",
#  ext_modules = cythonize(ext_modules)
#)
