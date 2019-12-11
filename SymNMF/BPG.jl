"""Computes Bregman distance D_h(A, B) = h(A) - h(B) - <grad h(B), A - B>
 associated to h(A) = 0.25 |A|^4 + alpha * 0.5 * |A|^2
"""
function Dh(A::Matrix{Float64}, B::Matrix{Float64}; alpha = 1., sigma = 1.)
    nA,nB = norm(A) ^ 2, norm(B) ^ 2
    hA = alpha * 0.25 * nA ^ 2 + sigma * 0.5 * nA
    hB = alpha * 0.25 * nB ^ 2 + sigma * 0.5 * nB
    gradh_B = (alpha * nB + sigma) * B
    return hA - hB - sum(gradh_B .* (A - B))
end

function grad_h(A::Matrix{Float64}; alpha = 1., sigma = 1.)
    return (alpha * norm(A) ^ 2 + sigma) * A
end

"""Bregman gradient step"""
function bregman_grad_step(A::Matrix{Float64}, grad::Matrix{Float64}, step::Float64 = 1., alpha::Float64 = 1., sigma::Float64 = 1.)

    z = alpha * norm(A) ^ 2 + sigma
    P = max.(z * A - step * grad, 0.)
    z = solve_cubic(alpha * norm(P) ^ 2, sigma)
    return P / z
end

"""Bregman proximal gradien update"""
function update_BPG(M::GenMatrix, A::Matrix{Float64}; step::Float64 = 1., alpha::Float64 = 1., sigma::Float64 = 1., kwargs...)
    grad = grad_SNMF(M, A)
    A_new = bregman_grad_step(A, grad, step, alpha, sigma)
    return A_new, norm(grad)
end

"""Checks the Nolips dynamical decrease condition
    f(A_new) <= f(A_old) + <grad f(A_old), A_new - A_old> + Dh(A_new, A_old) / step
"""
function bregman_decrease_cond(M::GenMatrix, A_new::Matrix{Float64},
    A_old::Matrix{Float64},
    grad::Matrix{Float64},
    old_loss::Float64,
    step::Float64;
    alpha::Float64 = 1., sigma::Float64 = 1.)

    new_loss = frobenius_sym_loss(A_new, M)
    lin = sum(grad .* (A_new - A_old))

    return new_loss - old_loss <= lin + Dh(A_new, A_old; alpha = alpha, sigma = sigma) / step
end

"""Bregman proximal gradient update with dynamic step size strategy"""
function update_BPGD(M::GenMatrix, A::Matrix{Float64}, step::Float64 = 1.; alpha::Float64 = 1., sigma::Float64 = 1., max_iter_ls::Int64 = 100., kwargs...)

    current_step = step
    grad = grad_SNMF(M, A)
    old_loss = frobenius_sym_loss(A, M)

    keep_going = true
    it = 0
    A_new = copy(A)

    while keep_going
        it += 1
        A_new = bregman_grad_step(A, grad, current_step, alpha, sigma)

        # checking decrease condition
        keep_going = !bregman_decrease_cond(M, A_new, A, grad, old_loss, current_step; alpha = alpha, sigma = sigma)
        keep_going = keep_going && (it <= max_iter_ls)

        current_step = current_step / 2
    end
    
    # computing the BB step size
#     delta_h4 = norm(A_new) * A_new
#     delta_h = grad_h(A_new; alpha = alpha) - grad_h(A; alpha = alpha)
#     delta_f = grad_SNMF(M, A_new) - grad
#     delta_A = A_new - A
#     acc_h = sum(delta_h .* delta_A) / norm(delta_A) ^ 2
#     acc_f = abs(sum(delta_f .* delta_A)) / norm(delta_A) ^ 2
#     current_step = acc_h / acc_f
#     current_step = abs(sum((delta_A) .* (delta_h)) / sum((delta_A) .* (delta_f)))
#     println("[step] $current_step  [ls_its] $it")
    
    # standard LS
    current_step = current_step * 4
    
#     println("\t\t [BB step size] ", current_step)
    
    return A_new, norm(grad), current_step
end

"""Bregman proximal gradient update with adaptive coefficient strategy"""
function update_BPGA(M::GenMatrix, A::Matrix{Float64}, step::Float64 = 1., alpha::Float64 = 1., sigma::Float64 = 1.; max_iter_ls::Int64 = 100, kwargs...)

    grad = grad_SNMF(M, A)
    old_loss = frobenius_sym_loss(A, M)

    keep_going = true
    it = 0
    current_step = step
    A_new = copy(A)

    while keep_going
        it += 1
        A_new = bregman_grad_step(A, grad, current_step, alpha, sigma)

        # checking decrease condition
        keep_going = !bregman_decrease_cond(M, A_new, A, grad, old_loss, current_step; alpha = alpha, sigma = sigma)
        keep_going = keep_going && (it <= max_iter_ls)

        current_step = current_step / 2
    end
    
    # computing the adaptive coefficients for the next iteration
    delta_A = A_new - A
    
    # 1) quartic part
    delta_h4 = norm(A_new) * A_new - norm(A) * A
    delta_f4 = 2 * A_new * (A_new' * A_new) - 2 * A * (A' * A)
    
    # 2) quadratic part
    delta_f2 = M * (A_new - A)
    
    #TODO : safeguard
    alpha = abs(sum((delta_A) .* (delta_h4)) / sum((delta_A) .* (delta_f4)))
    sigma = abs(sum((delta_A) .* (delta_A)) / sum((delta_A) .* (delta_f2)))
        
    println("[step] $current_step [alpha] $alpha [sigma] $sigma  [ls_its] $it")
    
    return A_new, norm(grad), current_step * 4, alpha, sigma
end


