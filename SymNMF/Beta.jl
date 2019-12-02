"""Beta-SNMF update.
Reference:
    Z. He, S. Xie, R. Zdunek et al. Symmetric Nonnegative Matrix Factorization:
    Algorithms and appliactions to probabistic Clustering.
    IEEE Transactions on Neural Networks, 2011
"""
function update_Beta(M::GenMatrix, A::Matrix{Float64}, E::Float64; beta = 1.)
    MA = M * A
    AAtA = A * (A' * A)
    R = MA ./ max.(AAtA, 1e-16)
    A_new = A .* (1. - beta .+ beta * R)
    E_new = frobenius_sym_loss(A, M)

    if E_new > E
        A_new = A .* (R .^ (1. / 3.))
        E_new = frobenius_sym_loss(A_new, M)
    end

    A = A_new
    E = E_new

    pgrad_norm = pgradnorm_SNMF(2 * (AAtA - MA), A)
    return A_new, pgrad_norm, E
end
