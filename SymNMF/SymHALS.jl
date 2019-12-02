"""Updates one block of variable for the SymHALS algorithm.
Efficient implementation is based on the FastHALS algorithm from
    A. Chichocki, A. Phan
    Fast local algorithms for large scale nonnegative matrix and tensor
        factorizations
    2009
"""
function update_A_symHALS(M::GenMatrix, A::Matrix{Float64}, Bt::Matrix{Float64},
    mu::Float64 = 1.)

    BBt = Bt' * Bt
    MBt = M * Bt

    for t = 1:size(BBt)[1]
        grad = - MBt[:,t] + A * BBt[t,:] + mu * (A[:,t] - Bt[:,t])

        # computing projected gradient ??
        hess = BBt[t,t] + mu

        if hess != 0
            A[:,t] = max.(A[:,t] - grad / hess, 0.)
        end
    end
end

"""SymHALS update for penalized SNMF algorithm.
Reference:
    Z. Zhu, X. Li et al
    Dropping Symmetry for Fast Symmetric Nonnegative Matrix Factorization
    NIPS, 2018
"""
function update_symHALS(M::GenMatrix, Mt::GenMatrix,
        A::Matrix{Float64}, Bt::Matrix{Float64}; mu = 1.)
    update_A_symHALS(M, A, Bt, mu)
    update_A_symHALS(Mt, Bt, A, mu)

    return A, Bt
end
