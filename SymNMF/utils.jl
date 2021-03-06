
using Distributions
using LinearAlgebra
using SparseArrays
using Hungarian

"""GenMatrix can be either a dense or a sparse CSC matrix"""
GenMatrix = Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int}}

"""Randomly initializes a matrix A for SymNMF of Matrix M with target rank r.
Sets the proper scaling"""
function random_init_sym(M::GenMatrix, r::Int)
    n = size(M)[1]
    scale = sqrt(mean(M) / float(r))

    A = 2. * rand(Uniform(1e-5, scale), n, r)
    return A
end

"""For a symmetric matrix M, computes
    |M|_{1,inf} = max( sum(M[i,j], j = 1..n), i = 1..n)
"""
function norm1inf(M::GenMatrix)
    return maximum(sum(M, dims = 1))
end

function frobenius_loss(A::Matrix{Float64}, B::Matrix{Float64}, M::GenMatrix)
    return 0.5 * (norm(A * B - M) ^ 2)
end

"""Efficient computation of the SymNMF loss function
    0.5 * |M - AAt| ^ 2
"""
function frobenius_sym_loss(A::Matrix{Float64}, M::GenMatrix, MA::Matrix{Float64})
   t1 = 0.5 * (norm(M) ^ 2 + norm(A' * A) ^ 2) # A' * A is a small r x r sized matrix
   
   mul!(MA, M, A)
   return t1 - dot(A, MA)
end

"""Same, but without preallocating MA
"""
function frobenius_sym_loss(A::Matrix{Float64}, M::GenMatrix)
   t1 = 0.5 * (norm(M) ^ 2 + norm(A' * A) ^ 2)
   
   t2 = dot(A, M * A)
   return t1 - t2
end

"""Computes the gradient of the SNMF objective function
    0.5 * |M - A * A'|^2
with preallocated M * A
"""
function grad_SNMF!(G::Matrix{Float64}, M::GenMatrix, A::Matrix{Float64}, MA::Matrix{Float64})
    mul!(MA, M, A)
    mul!(G, A, A' * A)
    @. G = 2 * (G - MA)
end

"""Computes the gradient of the SNMF objective function
    0.5 * |M - A * A'|^2
"""
function grad_SNMF(M::GenMatrix, A::Matrix{Float64})
    MA = M * A
    AAtA = A * (A' * A) # order is important for efficient computation
    return 2 * (AAtA - MA)
end

"""computes the squared norm of projected gradient of SNMF (useful for stopping criterion)
"""
function pgradnorm_SNMF(grad::Matrix{Float64}, A::Matrix{Float64})
    t1 = reduce(+, grad[i] ^ 2 for i in 1:length(A) if A[i] > 0.; init = 0.)
    t2 = reduce(+, min(0., grad[i]) ^ 2 for i in 1:length(A) if A[i] == 0.; init = 0.)
    t1 + t2
end

""" projection on nonnegative orthant"""
function project_orthant!(A::Array{Float64})

    for i = 1:length(A)
        if A[i] < 0.
            A[i] = 0.
        end
    end
end

"""computes the squared norm of the projected gradient of regular NMF
    f(A,B) = 0.5 * |AB - M| ^ 2
"""
function pgradnorm_NMF(M::GenMatrix, Mt::GenMatrix, A::Matrix{Float64}, Bt::Matrix{Float64};
    mu::Float64 = 1., kwargs...)
    grad_A = A * (Bt' * Bt) - M * Bt + mu * (A - Bt)
    grad_B = (A' * A) * Bt' - (Mt * A)' + mu * (Bt - A)'

    pgrad_A = pgradnorm_SNMF(grad_A, A) ^ 2
    pgrad_B = pgradnorm_SNMF(grad_B, Matrix(Bt')) ^ 2
    return pgrad_A + pgrad_B
end

function read_sparse_matrix(X_sp::Matrix{Float64})
    X = sparse(X_sp[1,:] .+ 1, X_sp[2,:] .+ 1, X_sp[3,:])
    return X
end

"""fins the unique real root of the equation
    z ^ 3 - sigma * z ^ 2 = c with c > 0
"""
function solve_cubic(c::Float64, sigma::Float64 = 1.)
    z = sigma / 3.
    sigma3 = sigma ^ 3
    delta = c ^ 2 + 4 * sigma3 * c / 27.
    sq_delta = sqrt(delta)

    b = 0.5 * c + sigma3 / 27.

    z = z + cbrt(b + 0.5 * sq_delta)
    z = z + cbrt(b - 0.5 * sq_delta)

    return z
end

function predict_clusters(A::Matrix{Float64})
    n = size(A, 1)

    predicted_clusters = Vector{Int64}(undef, n);

    for i = 1:n
       predicted_clusters[i] = findmax(A[i,:])[2]
    end

    return predicted_clusters
end

function clustering_accuracy(true_labels::Vector{Int64},
        predicted_labels::Vector{Int64},
        n_clusters::Int64)

    #building the cost matrix
    C = zeros(n_clusters, n_clusters)

    for i = 1:n_clusters, j = 1:n_clusters
        C[i,j] = sum((true_labels .== i) .& (predicted_labels .!= j))
    end

    assignment, cost = hungarian(C)
    return 1. - cost / length(true_labels)
end

"""Generates a synthetic sparse completely positive matrix of size (n,n) and rank r"""
function synthetic_SNMF(n::Int64, r::Int64; noise::Float64 = 0.5)
    A = rand(Bernoulli(1. / r), n, r)
    M = A * A' + noise * rand(Normal(), n, n)
    return 0.5 * max.(M + M', 0.)
end
