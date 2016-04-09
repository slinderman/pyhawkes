# Installing pyhawkes on OSX

This is a set of instructions for installing pyhawkes on OSX.

To install this requires compiling with OpenMP support, but OSX does not come with this support in its default `clang`. Thus you will need to install `clang-omp`:

    # install clang with OpenMP support (needed to compile later packages)
    brew update
    brew install clang-omp

Then, `gslrandom` is a dependency that requires OpenMP as well as a number of other packages included in the `requirements.txt`. To build `gslrandom` requires building it with the appropriate C and C++ compiler, which in this case will be `clang-omp` and `clang-omp++`. However, because it also requires other packages, namely `Cython` and `numpy` we will first install the requirements from `requirements.txt`:

    # install other dependencies
    pip install -r requirements.txt

Note because `gslrandom` requires a custom set of installation instructions though it is listed in the `requirements.txt` file it is commented out, and therefore is not automatically included in the `pip install` command. To install `gslrandom` we will install `gsl`, which is a non-python specific dependency and then will pip install `gslrandom` using `clang-omp` and `clang-omp++` as our `C` and `C++` compilers.

    # install gslrandom with OpenMP support
    brew install gsl
    CC=clang-omp CXX=clang-omp++ pip install gslrandom

Your environment should now be setup properly to either build a development or standard version of the library.  Depending on whether you want to build a development install or a standard install you will need to use one of the two following sets of commands:

    # if you want to develop the library, run:
    CC=clang-omp python setup.py build_ext --inplace
    CC=clang-omp pip install -e .

    # for a standard install, run:
    CC=clang-omp python setup.py install
