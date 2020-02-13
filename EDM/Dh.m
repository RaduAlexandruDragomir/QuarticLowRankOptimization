function output = Dh(X, Y, alpha, beta, sigma)
% output = Dh(X, Y, alpha, beta, sigma)
%   Computes the Bregman distance of the Gram quartic kernel between 
%  matrices X and Y with coefficients alpha, beta, sigma
%  
%   output : h(X) - h(Y) - < grad_h(Y), X-Y >
%
    hX = h_kernel(X, alpha, beta, sigma);
    hY = h_kernel(Y, alpha, beta, sigma);
    grad_h_Y = grad_h(Y, alpha, beta, sigma);
    output = hX - hY - sum(grad_h_Y .* (X - Y), 'all');
end

