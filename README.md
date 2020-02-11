# Quartic non-Euclidean algorithms for low rank minimization

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
