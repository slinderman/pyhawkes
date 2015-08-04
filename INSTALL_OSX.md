# installing pyhawkes on osx

    # install compiled dependencies
    brew update
    brew install clang-omp
    brew install gsl

    # install gslrandom with OpenMP support
    CC=clang-omp CXX=clang-omp++ pip install gslrandom

    # install other dependencies
    pip install -r requirements.txt

    # if you want to develop the library, run:
    CC=clang-omp python setup.py build_ext --inplace
    pip install -e .

    # for everyone else, run:
    CC=clang-omp python setup.py install
