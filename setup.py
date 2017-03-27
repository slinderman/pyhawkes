#!/usr/bin/env python

import os
import warnings
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

extra_compile_args = []
extra_link_args = []

# Only compile with OpenMP if user asks for it
USE_OPENMP = os.environ.get('USE_OPENMP', False)
if USE_OPENMP:
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
else:
    warnings.warn("Not using OpenMP for parallel parent updates. "
                  "This will incur a significant performance hit. "
                  "To compile with OpenMP support, make sure you are "
                  "using the GNU gcc and g++ compilers and then run "
                  "'export USE_OPENMP=True' before installing.")

ext_modules = cythonize('**/*.pyx')
for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)

setup(name='pyhawkes',
      version='0.2',
      description='Bayesian inference for network Hawkes processes',
      author='Scott Linderman',
      author_email='scott.linderman@gmail.com',
      url='http://www.github.com/slinderman/pyhawkes',
      ext_modules=ext_modules,
      install_requires=['numpy', 'scipy', 'matplotlib',
                        'joblib', 'scikit-learn', 'pybasicbayes'],
      include_dirs=[np.get_include(),],
      packages=['pyhawkes', 'pyhawkes.internals', 'pyhawkes.utils']
     )
