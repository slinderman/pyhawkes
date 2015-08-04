# installing pyhawkes on osx

This is a set of instructions for installing pyhawkes on OSX.

To install this requires compiling with openMP support, but OSX does not come with this support in its default clang. Thus you will need to install clang-omp.

    # install clang with OpenMP support (needed to compile later packages)
    brew update
    brew install clang-omp

Then, gslrandom is a dependency that requires OpenMP, and so to build it with the appropriate C and C++ compiler, which in this case will be clang-omp and clang-omp++.

    # install gslrandom with OpenMP support
    brew install gsl
    CC=clang-omp CXX=clang-omp++ pip install gslrandom

All other dependencies are python dependencies accessible through pip and the requirements.txt file.

    # install other dependencies
    pip install -r requirements.txt

Depending on whether you want to build a development install or a standard install you will need to use one of the two following sets of commands

    # if you want to develop the library, run:
    CC=clang-omp python setup.py build_ext --inplace
    pip install -e .

    # for a standard install, run:
    CC=clang-omp python setup.py install
