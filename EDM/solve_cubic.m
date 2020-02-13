function z = solve_cubic(c,alpha)
% fins the unique real root of the equation
%     z ^ 3 - alpha * z ^ 2 = c with c > 0

    z = alpha / 3.;
    alpha3 = alpha ^ 3;
    delta = c .^ 2 + 4 * alpha3 * c / 27.;
    sq_delta = sqrt(delta);
    
    b = 0.5 * c + alpha3 / 27.;
    
    z = z + nthroot(b + 0.5 * sq_delta,3);
    z = z + nthroot(b - 0.5 * sq_delta,3);
end

