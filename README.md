Fully Bayesian inference for discovering latent network structure underlying Hawkes processes. This work was 
 originally published in:
 
 Linderman, Scott W. and Adams, Ryan P. Discovering Latent Network Structure in Point Process Data. 
 *International Conference on Machine Learning (ICML)*, 2014.

To check out, run 
`git clone --recursive git@github.com:slinderman/pyhawkes.git`

To compile the cython code, run
`python setup.py build_ext --inplace`
  
This codebase corresponds to a new parameterization of the
Hawkes process model that will be outlined in a forthcoming paper.
It is considerably cleaner than the old CUDA version, and is still
pretty fast with the Cython+OMP extensions.
  