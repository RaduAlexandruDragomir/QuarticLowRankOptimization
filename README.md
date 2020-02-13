# Quartic first order algorithms for low rank minimization

We provide the code used for the numerical experiments in

> [1] R.A. Dragomir, A. d'Aspremont, J. Bolte, "Quartic First-Order Methods for Low Rank Minimization" arXiv preprint arXiv:1901.10791, 2019.


## 1. Symmetric Nonnegative Matrix Factorization

The [`SymNMF/`](SymNMF/) folder provides the experiments for Symmetric Nonnegative Matrix Factorization, implemented in [Julia](https://julialang.org).

### Getting started

It requires Julia 1.0.1+, as well as the packages

- Linear Algebra
- PyPlot
- Random
- SparseArrays
- IJulia and Jupyter (for running the notebook)

The experiments can be reproduced in the notebook [`SymNMF/Experiments.ipynb`](SymNMF/Experiments.ipynb)
All instructions to run the experiments for various datasets are detailed in the notebook.

### Contents

The folder [`data/`](data/) contains all preprocessed similarity matrices, for the 4 datasets used.

The code is organized as follows:
- [`SymNMF/SymNMF.jl`](/SymNMF/SymNMF.jl) : defines a wrapper for testing and monitoring all the algorithms. Algorithm-specific parameters can be passed as keyword arguments.
- [`SymNMF/BPG.jl`](SymNMF/BPG.jl) : implements Dyn-NoLips.
- [`SymNMF/Beta.jl`](SymNMF/Beta.jl) : implements the Beta-SNMF algorithm.
- [`SymNMF/CD.jl`](SymNMF/CD.jl): the coordinate descent algorithm
- [`SymNMF/data_preprocessing.jl`](SymNMF/data_preprocessing.jl): the tools for building similarity matrices. Data has already been preprocessed in our case
- [`SymNMF/NLS.jl`](SymNMF/NLS.jl) : the Block Principal Pivotting algorithm for solving Nonnegative
Least Squares problems for SymANLS. We used the code provided in [this package].(https://github.com/ahwillia/NonNegLeastSquares.jl).
- [`SymNMF/PG.jl`](SymNMF/PG.jl): the Projected Gradient algorithm with Armijo line search.
- [`SymNMF/SymANLS.jl`](SymNMF/SymANLS.jl) : implements the SymANLS penalty method.
- [`SymNMF/SymHALS.jl`](SymNMF/SymHALS.jl)SymHALS.jl : idem with SymHALS.
- [`SymNMF/utils.jl`](SymNMF/utils.jl) : useful functions related to SymNMF (initialization, evaluating the objective function...)


## 2. Euclidean Distance Matrix Completion

The [`EDM/`](EDM/) folder provides the code for the experiments on Euclidean Distane Matrix Completion (EDMC) problems.

The script [`EDM/main.m`](EDM/main.m) can be run to reproduce the experiments on Euclidean distance matrix completion with the synthetic Helix dataset.

The number of random initializations to average can be adjusted with the variable `n_runs`, and the number of data points with n. Note that, because of the variance induced by initial conditions, we advise do
at least 5 runs to get consistent results.

**Note:** Our implementation relies heavily on the code provided by Bamdev Mishra and Gilles Meyer that can be bound [here](https://bamdevmishra.in/codes/edmcompletion/), which is described in the following reference

> [2] B. Mishra, G. Meyer, and R. Sepulchre. Low-rank optimization for distance matrix completion. In Proceedings of the IEEE Conference on Decision and Control, 2011.

We used their implementation of Gradient Descent and Riemannian trust-region
algorithm as baselines. In order to run our experiments, we modified part of their code so as to monitor
the RMSE value across iterations. The modified code is in the [`EDM/code_Mishra2011`](EDM/code_Mishra2011) folder.

The Bregman Gradient/NoLips algorithm for EDMC is implemented in [`EDM/bg_dist_completion.m`](EDM/bg_dist_completion.m)
In order to do a fair comparison with the gradient descent implementation of Mishra et al. [2], we took the same line search and gradient computing procedures as in the function in [`EDM/code_Mishra2011/gd_dist_completion.m`](EDM/code_Mishra2011/gd_dist_completion.m).



