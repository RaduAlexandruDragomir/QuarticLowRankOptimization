"""Beta-SNMF update.
Reference:
    Z. He, S. Xie, R. Zdunek et al. Symmetric Nonnegative Matrix Factorization:
    Algorithms and appliactions to probabistic Clustering.
    IEEE Transactions on Neural Networks, 2011
"""
function update_Beta!(M::GenMatrix, A::Matrix{Float64}, A_new::Matrix{Float64}, MA::Matrix{Float64}, AAtA::Matrix{Float64}, G::Matrix{Float64}, E::Float64; beta = 1.)
 
    mul!(MA, M, A)
    mul!(AAtA, A, A' * A)
    
    @. AAtA = max.(AAtA, 1e-16)
    
    # tentative update
    @. A_new = (1 - beta) * A_new + beta .* MA ./ AAtA
 
    E_new = frobenius_sym_loss(A_new, M, G)

    # failure case
    if E_new > E
        @. A = A .* ( (MA ./ AAtA) .^ (1. / 3.))
        E_new = frobenius_sym_loss(A, M, G)
    else
        @. A = A_new
    end
    
    @. G = 2 * (AAtA - MA)
    pgrad_norm = pgradnorm_SNMF(G, A)

  return pgrad_norm, E_new
end
