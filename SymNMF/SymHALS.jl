"""Updates one block of variable for the SymHALS algorithm.
Efficient implementation is based on the FastHALS algorithm from
    A. Chichocki, A. Phan
    Fast local algorithms for large scale nonnegative matrix and tensor
        factorizations
    2009
"""
function update_A_symHALS!(M::GenMatrix, A::Matrix{Float64}, Bt::Matrix{Float64}, BBt::Matrix{Float64},
        MBt::Matrix{Float64}, grad_col::Vector{Float64}, grad::Matrix{Float64}, mu::Float64 = 1.)

    
    mul!(BBt, Bt', Bt)
    mul!(MBt, M, Bt)
    
    
    for t = 1:size(BBt)[1]
        BBt_t = @view BBt[t,:]
        mul!(grad_col, A, BBt_t)
        
        MBt_t = @view MBt[:,t]
        At = @view A[:,t] 
        Bt_t = @view Bt[:,t]
        @. grad_col = grad_col - MBt_t + mu * At - mu * Bt_t

        hess = BBt[t,t] + mu

        if hess != 0            
            for k = 1:size(A,1)
                A[k,t] = max(A[k,t] - grad_col[k] / hess, 0.)
            end
        end
    end
    
    # returning gradient norm for stopping criterion
    mul!(grad, Bt, BBt)
    @. grad = 2 * grad - 2 * MBt
    pgradnorm_SNMF(grad, Bt)
end

"""SymHALS update for penalized SNMF algorithm.
Reference:
    Z. Zhu, X. Li et al
    Dropping Symmetry for Fast Symmetric Nonnegative Matrix Factorization
    NIPS, 2018
"""
function update_symHALS!(M::GenMatrix, Mt::GenMatrix,
        A::Matrix{Float64}, Bt::Matrix{Float64}, BBt::Matrix{Float64},
        MBt::Matrix{Float64}, grad_col::Vector{Float64}, grad::Matrix{Float64}; mu = 1.)
    update_A_symHALS!(M, A, Bt, BBt, MBt, grad_col, grad, mu)
    pg_norm = update_A_symHALS!(Mt, Bt, A, BBt, MBt, grad_col, grad, mu)
    return pg_norm
end
