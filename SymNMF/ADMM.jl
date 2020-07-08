include("NLS.jl")

"""Update for ADMM algorithm for SymNMF.
from the following reference:
Reference:
    S. Lu, M. Hong, Z. Wang.
    A Nonconvex Splitting Method for Symmetric Nonnegative Matrix Factorization : Convergence Analysis and Optimality.
    IEEE Transactions on Signal Processing, 2017

Lambda is the dual matrix variable.
"""
function update_ADMM(M::GenMatrix, Mt::GenMatrix,
        A::Matrix{Float64}, Bt::Matrix{Float64}, Lambda::Matrix{Float64};
        rho::Float64 = 1.)

            # updating B
    gram_B = A' * A + rho * I
    grad_B = (Mt * A)' + rho * A' - Lambda

    Bt = Matrix(nnls_pivot(gram_B, grad_B, gram = true;
        X_init = Matrix(Bt'), tol = tol_inner)')

    # updating A
    gram_A = Bt' * Bt + rho * I
    grad_A = rho * Bt' + (M * Bt)' + Lambda
    A = Matrix(nnls_pivot(gram_A, grad_A, gram = true;
        X_init = Matrix(A'), tol = tol_inner)')

    # updating the multiplier
    Lambda = Lambda + rho * (Bt - A)
    
    return A, Bt, Lambda
end

