"""Generic wrapper for all SymNMF solvers.
Arguments

    - M::GenMatrix : the symmetric nonnegative matrix to factorize
    - r::Int: the target rank of the factorization

Keyword optionnal arguments
    - algo::Symbol : the algorithm to use.

        :pga : Projected Gradient method with Armijo line search
        :nolips : NoLips with fixed step size rule
        :dyn_nolips : NoLips with dynamical step size strategy
        :restart_nolips : NoLips with dynamical step size and
            Nesterov acceleration with restart.
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
        tol::Float64 = 1e-5,
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

    initial_pgnorm = max(pgradnorm_SNMF(grad_SNMF(M, A), A), 1e-16)
    pg_norm = initial_pgnorm

    losses = Array{Float64}(undef, 0, 4)
    t0 = time_ns()
    t_prev = t0

    keep_going = true
    pgnorm_cond = true
    it = 0


    # algorithm-specific initializations
    if algo == :pga
        step = kwargs[:step]

    elseif (algo == :nolips) || (algo == :dyn_nolips)  || (algo == :restart_nolips)
        alpha = norm1inf(M) / 3
        z = norm(A) ^ 2 + alpha
        initial_pgnorm = update_BPG(M, A, z; alpha = alpha, kwargs...)[2]

        if (algo == :restart_nolips)
            A_z = copy(A)
            tk = 1.
        end
        if (algo == :dyn_nolips) || (algo == :restart_nolips)
            step = kwargs[:step]
            max_step = 4. * r
        end

        if algo == :restart_nolips
            # restart strategy
            restart_count = 1 # alpha_restart^k
            C_restart, rho_restart = kwargs[:C_restart], kwargs[:rho_restart]
            restart_sched = C_restart
        end

    elseif algo == :beta
        E = frobenius_sym_loss(A, M)

    elseif algo == :cd
        C = sum(abs2, A, dims = 1)[:]
        L = sum(abs2, A, dims = 2)[:]
        D = A' * A
        A_coeffs = zeros(size(A))
        B_coeffs = zeros(size(A))

    elseif (algo == :sym_hals) || (algo == :sym_anls)
        Bt = copy(A)
        initial_pgnorm = pgradnorm_NMF(M, Mt, A, Bt; kwargs...)
        pg_norm = initial_pgnorm
    end

    while keep_going
        # monitoring loss
        if (monitoring_interval > 0.) && (float(time_ns() - t_prev) / 1e9 >= monitoring_interval) || (it == 0)
            delta_t = float(time_ns() - t_prev) / 1e9
            loss = frobenius_sym_loss(A, M)
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
            A, pg_norm, step = update_PG_armijo(M, A, step; kwargs...)
        elseif algo == :nolips
            A, pg_norm, z = update_BPG(M, A, z; alpha = alpha, kwargs...)
        elseif algo == :dyn_nolips
            A, pg_norm, z, step = update_BPGD(M, A, z, step;
                max_step = max_step, alpha = alpha, kwargs...)
        elseif algo == :beta
            A, pg_norm, E = update_Beta(M, A, E; kwargs...)
        elseif algo == :cd
            A, pg_norm, A_coeffs, B_coeffs, C, L, D = update_CD(M, A, A_coeffs, B_coeffs, C, L, D)
        elseif algo == :sym_hals
            A, Bt = update_symHALS(M, Mt, A, Bt; kwargs...)
            pg_norm = pgradnorm_NMF(M, Mt, A, Bt; kwargs...)
        elseif algo == :sym_anls
            A, Bt, = update_ANLS(M, Mt, A, Bt; kwargs...)
            pg_norm = pgradnorm_NMF(M, Mt, A, Bt; kwargs...)
        elseif algo == :restart_nolips
            A, pg_norm, A_z, z, tk, step = update_ABPGD(M, A, z, A_z, tk, step; alpha = alpha,
                max_step = max_step, kwargs...)

            # restart strategy
            if algo == :restart_nolips
                if restart_sched  == 0
                    A_z = copy(A)
                    tk = 1.
                    restart_count *= rho_restart
                    restart_sched = C_restart * restart_count
                else
                    restart_sched = restart_sched - 1
                end
            end
        end

        # checking stopping criterion
        it += 1
        time_cond = (time_ns() - t0) / 1e9 < max_time
        keep_going = (it <= max_iter) && time_cond

    end

    if (algo == :sym_hals) || (algo == :sym_anls)
       println("Constraint satisfaction for penalty method $algo : |A - B|/|A| = ",
        norm(A - Bt) / norm(A))
    end

    losses[:,1] = cumsum(losses[:,1])
    return A, losses
end
