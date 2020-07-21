"""Computes Bregman distance D_h(A, B) = h(A) - h(B) - <grad h(B), A - B>
 associated to h(A) = 0.25 |A|^4 + alpha * 0.5 * |A|^2
"""
function Dh(A::Matrix{Float64}, B::Matrix{Float64}; alpha = 1., sigma = 1.)
    nA,nB = norm(A) ^ 2, norm(B) ^ 2
    hA = alpha * 0.25 * nA ^ 2 + sigma * 0.5 * nA
    hB = alpha * 0.25 * nB ^ 2 + sigma * 0.5 * nB
    B_dot_AmB = dot(A, B) - nB
    return hA - hB - (alpha * nB + sigma) * B_dot_AmB
end

"""Bregman gradient step"""
function bregman_grad_step!(A::Matrix{Float64}, A_old::Matrix{Float64}, grad::Matrix{Float64}, step::Float64 = 1., alpha::Float64 = 1., sigma::Float64 = 1.)

    z = alpha * norm(A_old) ^ 2 + sigma
    @. A = z * A_old - step * grad
    project_orthant!(A)
 
    z = solve_cubic(alpha * norm(A) ^ 2, sigma)
    @. A = A / z
end

"""Bregman proximal gradien update"""
function update_BPG!(M::GenMatrix, A::Matrix{Float64}, grad::Matrix{Float64}, MA::Matrix{Float64}; step::Float64 = 1., alpha::Float64 = 1., sigma::Float64 = 1., kwargs...)
    grad_SNMF!(grad, M, A, MA)
    bregman_grad_step!(A, A, grad, step, alpha, sigma)
    return pgradnorm_SNMF(grad, A)
end

"""Checks the Nolips dynamical decrease condition
    f(A_new) <= f(A_old) + <grad f(A_old), A_new - A_old> + Dh(A_new, A_old) / step
"""
function bregman_decrease_cond(M::GenMatrix, A_new::Matrix{Float64},
    A_old::Matrix{Float64},
    grad::Matrix{Float64},
    MA::Matrix{Float64},
    old_loss::Float64,
    step::Float64;
    alpha::Float64 = 1., sigma::Float64 = 1.)

    new_loss = frobenius_sym_loss(A_new, M, MA)
    lin = dot(grad, A_new) - dot(grad, A_old)

    return new_loss - old_loss <= lin + Dh(A_new, A_old; alpha = alpha, sigma = sigma) / step
end


"""Bregman proximal gradient update with dynamic step size strategy"""
function update_BPGD!(M::GenMatrix, A::Matrix{Float64}, A_old::Matrix{Float64}, grad::Matrix{Float64}, MA::Matrix{Float64}, step::Float64 = 1.; alpha::Float64 = 1., sigma::Float64 = 1., max_iter_ls::Int64 = 100, kwargs...)

    current_step = step
 
    grad_SNMF!(grad, M, A, MA)
    old_loss = frobenius_sym_loss(A, M, MA)

    keep_going = true
    it = 0
    
    copy!(A_old, A)

    while keep_going
        it += 1
        
        bregman_grad_step!(A, A_old, grad, current_step, alpha, sigma)

        # checking decrease condition
        keep_going = !bregman_decrease_cond(M, A, A_old, grad, MA, old_loss, current_step; alpha = alpha, sigma = sigma)
  
        keep_going = keep_going && (it <= max_iter_ls)
        current_step = current_step / 2
    end

    pgradnorm_SNMF(grad, A), current_step * 4
end



