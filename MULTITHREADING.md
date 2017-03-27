# Installing pyhawkes with multithreading

This is a set of instructions for installing pyhawkes with high-performance multithreading.
For Mac OS X users, this assumes that you are using [Homebrew](https://brew.sh/) for package management.

## Install GSL and GNU compilers

For OS X users with Homebrew:

    # Install dependencies
    brew update
    brew install gsl
    brew install gcc --without-multilib

    # Make sure you're using GNU gcc and g++
    export CC="/usr/local/bin/gcc-x.x"   # <- replace with correct version
    export CXX="/usr/local/bin/g++-x.x"  # <- replace with correct version

If you're running Linux, make sure GSL and GCC are installed.

I haven't tried installing on Windows (if you do, please let me know!)

## Install gslrandom for parallel sampling of multinomial random variables

Now you can install `gslrandom` for multinomial resampling.

    # install other dependencies
    pip install gslrandom

## Install PyHawkes with OpenMP support

PyHawkes also (optionally) uses OpenMP for parallel parent updates of its continuous-time models.
If you're using GNU gcc and g++, you can install with OpenMP support as follows:

    # if you want to develop the library, run:
    USE_OPENMP=TRUE pip install -e .

    # for a standard install, run:
    USE_OPENMP=TRUE python setup.py install
