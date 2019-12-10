"""Projected gradient update with fixed step size"""
function update_PG(M::GenMatrix, A::Matrix{Float64}; step = 1.)
    grad = grad_SNMF(M, A)

    pgrad_norm = pgradnorm_SNMF(grad, A) # avoid re-computing gradient
    return max.(A - step * grad, 0.), pgrad_norm
end

"""checks the Armijo sufficient decrease condition"""
function sufficient_decrease_cond(M::GenMatrix, A_new::Matrix{Float64},
    A_old::Matrix{Float64},
    grad::Matrix{Float64},
    old_loss::Float64,
    sigma::Float64)

    treshold = sigma * sum(grad .* (A_new - A_old))
    new_loss = frobenius_sym_loss(A_new, M)
    return new_loss - old_loss <= treshold
end

"""Projected gradient update with improved Armijo line search.
Reference:
    C. Lin, Projected Gradient Methods for Nonnegative Matrix Factorization,
    Neural Computation, 2007.
"""
function update_PG_armijo(M::GenMatrix, A::Matrix{Float64},
        current_step::Float64 = 1.;
        beta::Float64 = 0.1,
        sigma::Float64 = 0.01,
        max_inner_iter::Int = 20, kwargs...)

    keep_going = true
    it = 0

    step = current_step
    grad = grad_SNMF(M, A)
    old_loss = frobenius_sym_loss(A, M)
    A_temp = max.(A - step * grad, 0.)

    while keep_going
        step = step * beta
        A_temp = max.(A - step * grad, 0.)
        suff_decrease = sufficient_decrease_cond(M, A_temp, A, grad, old_loss, sigma)

        it += 1
        keep_going = !suff_decrease && (it <= max_inner_iter)
    end

#     println("PG  [it] ", it, " [eff_step_size] ", step)
    pgrad_norm = pgradnorm_SNMF(grad, A)
    return A_temp, pgrad_norm, step / beta
end
