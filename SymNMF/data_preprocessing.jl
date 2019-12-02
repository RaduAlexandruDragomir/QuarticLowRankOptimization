using Distances

"""
Sparsifies a graph adjacency matrix D to keep only the k-nearest neighbors of each
point.
"""
function sparsify_graph(D::GenMatrix, k::Int)
    knns = zeros(size(D))

    for i = 1:size(D)[1]
        i_knns = partialsortperm(D[i,:], 1:k, rev = true)
        knns[i, i_knns] = knns[i, i_knns] .+ 1.
    end

    sym_knn = min.(knns + knns', 1.)
    sim_sparsified = D .* sym_knn

    # 3) normalizing
    norm_factor = sqrt.(sum(sim_sparsified, dims = 1))
    return sim_sparsified ./ (norm_factor' * norm_factor)
end

"""Builds a similarity matrix from text data X. k controls the number of nearest
neighbors to keep at the sparsification phase.
The procedure is the one detailed in Section 7.1. of
D. Kuang, S. Yun, H.Park
SymNMF: nonnegative low-rank approximation of a similarity matrix for graph clustering
Journal of Global Optimization 2015
"""
function sim_matrix_text(X::GenMatrix, k::Int)
    # 1) constructing the graph

    # normalizing the matrix
    row_norms = sqrt.(sum(abs2, X, dims = 2))

    X_normed = X ./ row_norms

    # cosine similarity
    cosine_sim = X_normed * X_normed'

    # 2) sparsifying the graph
    return sparsify_graph(cosine_sim, k)
end

"""Builds a similarity matrix from image data X. k controls the number of nearest
neighbors to keep at the sparsification phase.
The procedure is the one detailed in Section 7.1. of
D. Kuang, S. Yun, H.Park
SymNMF: nonnegative low-rank approximation of a similarity matrix for graph clustering
Journal of Global Optimization 2015
"""
function sim_matrix_img(X::GenMatrix, k::Int)
    distances = pairwise(Euclidean(), X)
    n = size(distances)[1]

    # computing local scales
    sigma = zeros(n)

    scale_rank = 7
    for i=1:n
        k_nn = partialsortperm(distances[i,:], scale_rank)
        sigma[i] = distances[i,k_nn]
    end

    ## dont forget squared!!!
    norm_distances = (- distances .^ 2) ./ (sigma * sigma')
    gauss_distances = exp.(norm_distances)

    if k > 0
       gauss_distances = sparsify_graph(gauss_distances, k)
    end

    return gauss_distances
end
