
function [Y infos] = bg_dist_completion(Is,Js,knownDists,Y,trueDists,params)
% [Y infos] = bg_dist_completion(Is,Js,knownDists,Y,trueDists,params)
%
% Given (i,j,d)'s contained in [Is,Js,knownDists], the function compute a
% scalar product matrix X = Y*Y' that minimizes the mean quadratic error on
% observed distances. i and j's value range from 1 to n, the total number
% of points.
% Uses the Bregman Gradient/NoLips algorithm with a quartic kernel to
% solve the minimization problem.
% 
% The code has been adapted from the Gradient descent algorithm of
% Bamdev Mishra and Gilles Meyer
% implemented in code_Mishra2011/gd_dist_completion.m
%
%   The parameters and output are the same as for gd_dist_completion,
%   with additional options in the field params specific to NoLips:
%      - params.alpha, params.beta, params.sigma : coefficients of the
%      quartic Bregman kernel 
%      - params.gram_inner_tol : tolerance for the inner loop when using
%       the Gram kernel (i.e. when params.beta > 0)
%      -params.gram_max_iter : maximal number of iterations in the inner
%      loop when using the Gram kernel (i.e. when params.beta > 0)
%   
%

% Problem size
[n,r] = size(Y);

% Data size
m = length(knownDists);

% Collect parameters
params = DefaultParams(params);

% Shortcut parameter names
maxiter = params.maxiter;
sig_A = params.sig_A;
ls_maxiter = params.ls_maxiter;
max_step = params.max_step;
tol = params.tol;
vtol = params.vtol;
verb = params.verb;

monitor_rmse = params.monitor_rmse;
monitor_interval = params.monitor_interval;

alpha = params.alpha;
beta = params.beta;
sigma = params.sigma;

gram_knl = beta > 0.;

% Fixed matrix that results from the gradient.
% For efficiency, it is computed once and for all at the beginning.
if isfield(params,'EIJ'),
    EIJ = params.EIJ;
else
    EIJ = speye(n);
    EIJ = EIJ(:,Is) - EIJ(:,Js);
end

% Compute initial cost
Z = EIJ'*Y;
estimDists = sum(Z.^2,2);
errors = (estimDists - knownDists);
mean_knownDists = mean(knownDists .^ 2);

S = sparse(1:m,1:m,2 * errors / m,m,m,m);
grad_init = EIJ * S * Z;
gradientNorm_init = norm(grad_init,'fro')^2;

cost = mean(errors.^2);

if verb,
    fprintf('[%0.4d] cost = %g\n',0,cost);
end

infos.costs = zeros(maxiter+1,1);
infos.costs(1) = cost;
infos.sqnormgrad = [];
infos.linesearch = [];
infos.time = [0.];
infos.rmse = [RMSE(Y,trueDists)];
infos.rmse_time = [0.];

time_init = cputime;

mu = zeros(r,1);

% Loop through all iterations
for iter = 1 : maxiter,
    
    S = sparse(1:m,1:m,2 * errors / m,m,m,m);
    
    % Compute gradient
    grad_Y = EIJ * S * Z;
    gradientNorm = norm(grad_Y,'fro')^2;
    infos.sqnormgrad = [infos.sqnormgrad ; gradientNorm];
    
    if iter > 1,  
        % Adaptive stepsize search as proposed by Mishra et al., arXiv:1209.0430
        if s == 1,
            max_step = 2*max_step;
        elseif s >= 3,
            max_step = 2*stepSize;
        else % here s == 2,
            max_step = max_step; % Do nothing 
        end
        
    end
    
    % Perform line-search
    Yt = Y;
    stepSize = max_step;

    for s = 1 : ls_maxiter,
        
        % bregman step
        P = grad_h(Yt, alpha, beta, sigma) - stepSize * grad_Y;
        
        if gram_knl
            [Y, mu] = grad_hg_star(P, alpha, beta, sigma, mu, params);
        else
            Y = grad_h_star(P, alpha, beta, sigma);
        end
            
        % Evaluate new cost
        Z = EIJ'*Y;
        estimDists = sum(Z.^2,2);
        errors = (estimDists - knownDists);
        
        newcost = mean(errors.^2);
        
        % Check Armijo condition
        diff_cost = newcost - cost;
        lin_part = sum(grad_Y .* (Y - Yt), 'all');
        diff_part = Dh(Y,Yt,alpha, beta,sigma);
        armijo = diff_cost <= lin_part + diff_part / stepSize;
        if armijo,
            break;
        else
            stepSize = stepSize / 2;
        end
        
    end
    
    % centering the matrix
    Y = Y - mean(Y,1);
    
    if mod(iter,monitor_interval) == 0
        current_rmse = RMSE(Y,trueDists);
        infos.rmse_time = [infos.rmse_time; cputime - time_init];
        infos.rmse = [infos.rmse; current_rmse];
    end
    
    infos.costs(iter+1) = newcost;
    infos.time = [infos.time; cputime - time_init];
    
    infos.linesearch= [infos.linesearch; s];
    if verb,
        fprintf('[BG][%0.4d] cost = %g, grad norm = %g, #linesearch = %g, stepsize = %g\n',iter,newcost,gradientNorm, s, stepSize);
    end
    
    % Stopping criterion
    if (gradientNorm / gradientNorm_init) < tol || abs(cost - newcost)/cost < vtol,
        fprintf('Stopping criterion reached\n');
        break;
    end
    
    cost = newcost;
    
end


infos.costs = infos.costs(1:iter+1);
infos.iter = iter;

if iter >= maxiter,
    warning('MATLAB:MaxIterReached','Maximum number of iterations has been reached.');
end

end