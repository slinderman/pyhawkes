#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='pyhawkes',
      version='0.1',
      description='Bayesian inference for network Hawkes processes',
      author='Scott Linderman',
      author_email='slinderman@seas.harvard.edu',
      url='http://www.github.com/slinderman/pyhawkes',
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
     )