#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

extra_compile_args = ['-fopenmp']
extra_link_args    = ['-fopenmp']

ext_modules = cythonize('**/*.pyx')
for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)

setup(name='pyhawkes',
      version='0.1',
      description='Bayesian inference for network Hawkes processes',
      author='Scott Linderman',
      author_email='scott.linderman@columbia.edu',
      url='http://www.github.com/slinderman/pyhawkes',
      ext_modules=ext_modules,
      include_dirs=[np.get_include(),],
      packages=['pyhawkes', 'pyhawkes.internals', 'pyhawkes.utils']
     )
