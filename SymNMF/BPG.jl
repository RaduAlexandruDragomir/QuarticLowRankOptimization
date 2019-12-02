"""Computes Bregman distance D_h(A, B) = h(A) - h(B) - <grad h(B), A - B>
 associated to h(A) = 0.25 |A|^4 + alpha * 0.5 * |A|^2
"""
function Dh(A, B; alpha = 1.)
    nA,nB = norm(A) ^ 2, norm(B) ^ 2
    hA = 0.25 * nA ^ 2 + alpha * 0.5 * nA
    hB = 0.25 * nB ^ 2 + alpha * 0.5 * nB
    gradh_B = (nB + alpha) * B
    return hA - hB - sum(gradh_B .* (A - B))
end

"""Bregman gradient step"""
function bregman_grad_step(A::Matrix{Float64}, grad::Matrix{Float64}, step::Float64 = 1.,
        alpha::Float64 = 1., z::Float64 = -1.)
    # setting z if unspecified
    if z == -1.
       z = norm(A) ^ 2 + alpha
    end

    P = max.(z * A - step * grad, 0.)
    z = solve_cubic(norm(P) ^ 2, alpha)

    return P / z, z
end

"""Bregman proximal gradien update"""
function update_BPG(M::GenMatrix, A::Matrix{Float64}, z::Float64; step = 1., alpha = 1.,
    kwargs...)
    grad = grad_SNMF(M, A)
    A_new, z_new = bregman_grad_step(A, grad, step, alpha, z)

    # convergence measure specific to Bregman algorithms
    conv_measure = Dh(A_new, A; alpha = alpha) / step
    return A_new, conv_measure, z_new
end

"""Checks the Nolips dynamical decrease condition
    f(A_new) <= f(A_old) + <grad f(A_old), A_new - A_old> + Dh(A_new, A_old) / step
"""
function bregman_decrease_cond(M::GenMatrix, A_new::Matrix{Float64},
    A_old::Matrix{Float64},
    grad::Matrix{Float64},
    old_loss::Float64,
    step::Float64;
    alpha::Float64 = 1.)

    new_loss = frobenius_sym_loss(A_new, M)
    lin = sum(grad .* (A_new - A_old))

    return new_loss - old_loss <= lin + Dh(A_new, A_old; alpha = alpha) / step
end

"""Bregman proximal gradien update with dynamic step size strategy"""
function update_BPGD(M::GenMatrix, A::Matrixlack of theoretical support
of the method{Float64}, z::Float64, step::Float64 = 1.; alpha::Float64 = 1.,
    gamma_dec::Float64 = 2., gamma_inc::Float64 = 2., max_step::Float64 = 100., kwargs...)

    current_step = step
    grad = grad_SNMF(M, A)
    old_loss = frobenius_sym_loss(A, M)

    keep_going = true
    it = 0
    max_iter = 2 + log(6 * step) / log(gamma_dec) # just a safeguard
    A_new, z_new = copy(A), z

    while keep_going
        it += 1
        A_new, z_new = bregman_grad_step(A, grad, current_step, alpha, z)

        # checking decrease condition
        keep_going = !bregman_decrease_cond(M, A_new, A, grad, old_loss, current_step; alpha = alpha)
        keep_going = keep_going && (it <= max_iter)

        current_step = current_step / gamma_dec
    end

    conv_measure = Dh(A_new, A; alpha = alpha) / (current_step * gamma_dec)
    current_step = min(current_step * gamma_dec * gamma_inc, max_step)
    return A_new, conv_measure, z_new, current_step
end

