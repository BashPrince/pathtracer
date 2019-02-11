from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize([Extension("PyRenderer",
                                        ["renderer/cython/PyRenderer.pyx"],
                                        libraries = ["renderer"],
                                        library_dirs=["renderer/cpp/lib/"])]))