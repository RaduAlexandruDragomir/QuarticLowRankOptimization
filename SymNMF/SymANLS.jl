include("NLS.jl")

"""Update for SymANLS algorithm.
Reference:
    D. Kuang, S. Yun, H. Park
    SymNMF: nonnegative low-rank approximation of a similarity matrix for graph clustering.
    Journal of Global Optimization, 2015
"""
function update_ANLS(M::GenMatrix, Mt::GenMatrix,
        A::Matrix{Float64}, Bt::Matrix{Float64};
        mu::Float64 = 1., tol_inner::Float64 = 1e-10)

            # updating B
    gram_B = A' * A + mu * I
    grad_B = (Mt * A)' + mu * A'

    Bt = Matrix(nnls_pivot(gram_B, grad_B, gram = true;
        X_init = Matrix(Bt'), tol = tol_inner)')

    # updating A
    gram_A = Bt' * Bt + mu * I
    grad_A = mu * Bt' + (M * Bt)'
    A = Matrix(nnls_pivot(gram_A, grad_A, gram = true;
        X_init = Matrix(A'), tol = tol_inner)')

    return A, Bt
end
