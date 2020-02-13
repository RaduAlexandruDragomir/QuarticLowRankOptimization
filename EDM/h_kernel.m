function output = h_kernel(X, alpha, beta, sigma)
% output = h_kernel(X, alpha, beta, sigma)
% Computes the value of the quartic kernel with coefficients
% alpha, beta, sigma applied to matrix X
% 

    nX = norm(X, 'fro')^2;
    output = 0.25 * alpha * nX^2 + 0.5 * sigma * nX;
    
    if beta > 0.
       output = output + 0.25 * beta * norm(X' * X, 'fro')^2;
    end
end

