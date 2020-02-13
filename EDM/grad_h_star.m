function output = grad_h_star(P, alpha, beta, sigma)
% output = grad_h(P, alpha, beta, sigma)
% Computes the mirror map associated to the general quartic kernel between 
% matrices X and Y with coefficients alpha, beta, sigma,
% i.e., the solution to the problem
%   min_X h(X) - <P,X>  
%  
%   if beta = 0, this reduces to the norm quartic kernel.

    if beta == 0.
        z = solve_cubic(norm(P, 'fro')^2 * alpha, sigma);
        output = P / z;
    else
        output = grad_hg_star(P, alpha, beta, sigma);
    end
end