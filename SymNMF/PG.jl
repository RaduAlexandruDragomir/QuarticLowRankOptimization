"""Projected gradient update with fixed step size"""
function update_PG!(M::GenMatrix, A::Matrix{Float64}, grad::Matrix{Float64}, MA::Matrix{Float64}; step = 1.)
    grad_SNMF!(grad, M, A, MA)

    pgrad_norm = pgradnorm_SNMF(grad, A) # avoid re-computing gradient
    @. A = A - step * grad
    project_orthant!(A)
    return pgrad_norm
end

"""checks the Armijo sufficient decrease condition"""
function sufficient_decrease_cond(M::GenMatrix, A_new::Matrix{Float64},
    A_old::Matrix{Float64},
    grad::Matrix{Float64}, 
    MA::Matrix{Float64},
    old_loss::Float64,
    sigma::Float64)

    treshold = sigma * (dot(grad, A_new) - dot(grad, A_old))
    new_loss = frobenius_sym_loss(A_new, M, MA)
    return new_loss - old_loss <= treshold
end

"""Projected gradient update with improved Armijo line search.
Reference:
    C. Lin, Projected Gradient Methods for Nonnegative Matrix Factorization,
    Neural Computation, 2007.
"""
function update_PG_armijo!(M::GenMatrix, A::Matrix{Float64}, A_old::Matrix{Float64}, grad::Matrix{Float64}, 
  MA::Matrix{Float64},
        current_step::Float64 = 1.;
        beta::Float64 = 0.1,
        sigma::Float64 = 0.01,
        max_inner_iter::Int = 20, kwargs...)

    keep_going = true
    it = 0

    step = current_step
    grad_SNMF!(grad, M, A, MA)
    old_loss = frobenius_sym_loss(A, M, MA)
 
    copy!(A_old, A)
 
    
    while keep_going
        step = step * beta
  
        @. A = A_old - step * grad
        project_orthant!(A)

        suff_decrease = sufficient_decrease_cond(M, A, A_old, grad, MA, old_loss, sigma)

        it += 1
        keep_going = !suff_decrease && (it <= max_inner_iter)
    end

    pgrad_norm = pgradnorm_SNMF(grad, A)
    pgrad_norm, step / beta^2
end
