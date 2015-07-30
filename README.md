Fully Bayesian inference for discovering latent network structure underlying Hawkes processes. This work was 
 originally published in:
 
 Linderman, Scott W. and Adams, Ryan P. Discovering Latent Network Structure in Point Process Data. 
 *International Conference on Machine Learning (ICML)*, 2014.

Examples
===

![True Model](https://raw.githubusercontent.com/slinderman/pyhawkes/master/data/gifs/true.gif)
![Inferred Model](https://raw.githubusercontent.com/slinderman/pyhawkes/master/data/gifs/hawkes_inf_anim.gif)


Installation
===
To check out, run 
`git clone --recursive git@github.com:slinderman/pyhawkes.git`

To compile the cython code, run
`python setup.py build_ext --inplace`
  
This codebase is considerably cleaner than the old CUDA version, and is still
quite fast with the Cython+OMP extensions and joblib for parallel sampling of
the adjacency matrix.
  