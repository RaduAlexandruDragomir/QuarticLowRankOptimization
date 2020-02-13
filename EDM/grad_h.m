function output = grad_h(X, alpha, beta, sigma)
% output = grad_h(X, alpha, beta, sigma)
%   Computes the gradient of the Gram quartic kernel between 
%  matrices X and Y with coefficients alpha, beta, sigma
%  

    n = norm(X, 'fro')^2;
    
    output = (alpha * n + sigma) * X;
    
    if sigma > 0.
       output = output + beta * X * (X' * X);  
    end
end

