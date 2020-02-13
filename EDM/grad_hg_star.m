function [output,mu] = grad_hg_star(V,alpha,beta,sigma,mu_init,params)
% grad_hg_star(V,alpha,beta,sigma,mu_init,params)
% Computes the gradient of the conjugate of the Gram kernel
% Uses NoLips with the universal quartic kernel for the inner minimization
%        Parameters:
%           - V : input matrix
%           - alpha,beta,sigma: the coefficients of the Gram kernel
%           - mu_init : previous value for warm starting mu
%       
%       Output : the value of the gradient of hg* applied to V

    % eigdecomposition of V^T V
    gramV = V' * V;
    [P,D] = eig(gramV);
    eta = sqrt(diag(D));
    
    inner_tol = params.gram_inner_tol;
    max_it = params.gram_max_iter;
    
    mu = mu_init;
    alpha_u = alpha + 3 * beta;
    sigma_u = sigma;
    
    if alpha > 0
        for it = 1:max_it
            n_mu = norm(mu)^2;

            grad_phi = alpha * n_mu * mu + beta * mu .^ 3;
            grad_phi = grad_phi + sigma * mu - eta;
            grad_h = (alpha + 3 * beta) * n_mu * mu + sigma * mu;
            v = grad_h - grad_phi;
            z = solve_cubic(norm(v)^2 * (alpha + 3 * beta), sigma);

            mu = v / z;

            % stopping criterion
            if norm(grad_phi) / norm(eta) < inner_tol
                break;
            end
        end

        Z_eigvals = mu .^ 2;
        diag_inv = beta * Z_eigvals + alpha * sum(Z_eigvals) + sigma;
        fac_inv = P * diag(1 ./ diag_inv) * P';
        output = V * fac_inv;
    else
        eig_z = solve_cubic(diag(D) * beta, sigma);
        fac_inv = P * diag(1 ./ eig_z) * P';
        output = V * fac_inv;
    end
end

