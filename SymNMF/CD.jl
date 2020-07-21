function poly4(a::Float64, b::Float64, x::Float64)
    return 0.25 * x^4 + a * 0.5 *  x ^ 2 + b * x
end

function cubic_root(x)
    x = real(x)
   if x >= 0.
        return x ^ (1. / 3.)
    else
        return - (abs(x) ^ (1. / 3.))
    end
end

"""Returns argmin_{x >= 0} x^4 / 4 + a x ^ 2 / 2 + b x
"""
function best_poly_root(a::Float64, b::Float64)
    delta = 4 * a ^ 3 + 27 * b ^ 2
    d = 0.5 * (-b + sqrt(Complex(delta / 27.)))

    if delta <= 0.
        r = 2 * cubic_root(abs(d))
        theta = angle(d) / 3.
        best_z = 0.
        best_y = 0.

        for k = 0:2
            z = r * cos(theta + 2 * k * pi / 3.)
            if (z >= 0.) && (poly4(a, b, z) < best_y)
                best_z = z
                best_y = poly4(a, b, z)
            end
        end

        return best_z
    else
        z_c = cubic_root(d) + cubic_root(0.5 * (- b - sqrt(delta / 27.)))
        z = Real(z_c)
        if (z >= 0.) && (poly4(a, b, z) < 0.)
            return z
        else
            return 0.
        end
    end
end

   
   
"""Does a full iteration of coordinate descent SymNMF.
Reference:
    A. Vandaele, N.Gillis et al.
    Efficient and non-convex coordinate descent for symmetric nonnegative matrix factorization.
    IEEE Transactions on Signal Processing, 2016
"""
function update_CD!(M::GenMatrix, A::Matrix{Float64}, MA::Matrix{Float64}, grad::Matrix{Float64},
    A_coeffs::Matrix{Float64}, B_coeffs::Matrix{Float64},
    C::Vector{Float64}, L::Vector{Float64}, D::Matrix{Float64})

    n, r = size(A)
    
    for j = 1:r
        for i = 1:n
            A_coeffs[i,j] = C[j] + L[i] - 2. * A[i,j] ^ 2 - M[i,i]
          
            Ai = @view(A[i,:])
            Dj = @view(D[:,j])
            Aj = @view(A[:,j])
            Mi = @view(M[:,i])
            B_coeffs[i,j] = dot(Ai, Dj) - dot(Aj, Mi)
            B_coeffs[i,j] = B_coeffs[i,j] - A[i,j] ^ 3 - A[i,j] * A_coeffs[i,j]
            Aij_new = best_poly_root(A_coeffs[i,j], B_coeffs[i,j])

            C[j] = C[j] + Aij_new ^ 2 - A[i,j] ^ 2
            L[i] = L[i] + Aij_new ^ 2 - A[i,j] ^ 2

            @. D[j,:] = D[j,:] + A[i,:] * (Aij_new - A[i,j])
            @. D[:,j] = D[j,:]
            D[j,j] = C[j]
            A[i,j] = Aij_new

        end
    end

    grad_SNMF!(grad, M, A, MA)
    pg_norm = pgradnorm_SNMF(grad, A)
    return pg_norm
end
