"""Generic wrapper for all SymNMF solvers.
Arguments

    - M::GenMatrix : the symmetric nonnegative matrix to factorize
    - r::Int: the target rank of the factorization

Keyword optionnal arguments
    - algo::Symbol : the algorithm to use.

        :pga : Projected Gradient method with Armijo line search
        :nolips : NoLips with fixed step size rule
        :dyn_nolips : NoLips with dynamical step size strategy
        :adap_nolips : NoLips with adaptive coefficients
        :beta : Beta-SNMF
        :cd : coordinate descent
        :sym_hals
        :sym_anls

    - max_iter::Int : maximal number of iterations
    - max_iter::Float64 : maximal running time (seconds)
    - monitoring_interval::Float64 : the interval between evaluations of objective
        and/or clustering accuracy. Set to 0 for no monitoring

    - A_init::GenMatrix : initial value of matrix A
    - monitor_accuracy : set to true to measure the clustering accuracy with respect
        to the true labels
    - true_labels::Vector{Int64} : ground truth labels of the dataset

Optionnal algorithm-specific parameters can also be passed via keyword arguments.s

Output
    - A::Matrix{Float64} : the solution matrix A
    - losses::Matrix{Float64} : a (n_measures, 4) size matrix where
        losses[:,1] is the time of the measures
        losses[:,2] are the values of the objective function
        losses[:,3] the convergence measure
        losses[:,4] the clustering accuracies
"""
function SymNMF(M::GenMatrix, r::Int;
        algo::Symbol = :bpg,
        max_iter::Int = 500, max_time::Float64 = 60.,
        tol::Float64 = 1e-4,
        monitoring_interval = 0.,
        A_init::GenMatrix = nothing,
        monitor_accuracy = false,
        true_labels::Vector{Int64} = nothing,
        kwargs...)

        n = size(M, 1)

    # initialization
    if A_init == nothing
        A = random_init_sym(M, r)
    else
        A = copy(A_init)
    end

    # for faster computations we store also the matrix M'
    if typeof(M) == SparseMatrixCSC{Float64, Int}
        Mt = sparse(M')
    else
        Mt = Matrix(M')
    end
   
    MA = zeros(size(A))
    grad = zeros(size(A))
   
    grad_SNMF!(grad, M, A, MA)
    initial_pgnorm = max(pgradnorm_SNMF(grad, A), 1e-16)
    pg_norm = initial_pgnorm

    losses = Array{Float64}(undef, 0, 4)
    t0 = time_ns()
    t_prev = t0

    keep_going = true
    pgnorm_cond = true
    it = 0

    # algorithm-specific initializations
    if (algo == :pga) || (algo == :nolips) || (algo == :dyn_nolips)
        step = kwargs[:step]
        
        if (algo == :nolips) || (algo == :dyn_nolips)
            alpha = 6.
            sigma = 2*norm(M)
        end
    
        A_old = copy(A)

    elseif algo == :beta
        E = frobenius_sym_loss(A, M, MA)
        A_new = zeros(size(A))
        AAtA  = zeros(size(A))

    elseif algo == :cd
        C = sum(abs2, A, dims = 1)[:]
        L = sum(abs2, A, dims = 2)[:]
        D = A' * A
        A_coeffs = zeros(size(A))
        B_coeffs = zeros(size(A))

    elseif (algo == :sym_hals) || (algo == :sym_anls) || (algo == :admm)
        Bt = copy(A)
        
        MBt = zeros(size(A))
        BBt = zeros(r,r)
        grad_col = zeros(n)

        if algo == :admm
            Lambda = zeros(size(A))
        end
    end

    while keep_going
        # monitoring loss
        if (monitoring_interval > 0.) && (float(time_ns() - t_prev) / 1e9 >= monitoring_interval) || (it == 0)
            delta_t = float(time_ns() - t_prev) / 1e9
            loss = frobenius_sym_loss(A, M, MA)
            clust_acc = 0.

            if monitor_accuracy
                y_pred = predict_clusters(A)
                clust_acc = clustering_accuracy(true_labels, y_pred, r)
            end

            losses = vcat(losses, [delta_t loss (pg_norm / initial_pgnorm) clust_acc])
            t_prev = time_ns()
        end

        # doing the update specific to each algorithm
        if algo == :pga
            pg_norm, step = update_PG_armijo!(M, A, A_old, grad, MA, step; kwargs...)
        elseif algo == :nolips
            pg_norm = update_BPG!(M, A, grad, MA; kwargs...)
        elseif algo == :dyn_nolips
            pg_norm, step = update_BPGD!(M, A, A_old, grad, MA, step; alpha = alpha, sigma = sigma, kwargs...)
        elseif algo == :beta
            pg_norm, E = update_Beta!(M, A, A_new, MA, AAtA, grad, E; kwargs...)
        elseif algo == :cd
            pg_norm = update_CD!(M, A, MA, grad, A_coeffs, B_coeffs, C, L, D)
        elseif algo == :sym_hals
            pg_norm = update_symHALS!(M, Mt, A, Bt, BBt, MBt, grad_col, grad; kwargs...)
        elseif algo == :sym_anls
            update_ANLS!(M, Mt, A, Bt; kwargs...)
            grad_SNMF!(grad, M, A, MA)
            pg_norm = pgradnorm_SNMF(grad, A)
        elseif algo == :admm
            update_ADMM!(M, Mt, A, Bt, Lambda; kwargs...)
            grad_SNMF!(grad, M, A, MA)
            pg_norm = pgradnorm_SNMF(grad, A)
        end

        # checking stopping criterion
        it += 1
        time_cond = (time_ns() - t0) / 1e9 < max_time
        keep_going = (it <= max_iter) && time_cond && pg_norm / initial_pgnorm > tol

    end

#     if (algo == :sym_hals) || (algo == :sym_anls) || (algo == :admm)
#        println("Constraint satisfaction for penalty method $algo : |A - B|/|A| = ",
#         norm(A - Bt) / norm(A))
#     end
                                                        
#     println("Terminated after $it iterations.")
    losses[:,1] = cumsum(losses[:,1])
    return A, losses 
end
