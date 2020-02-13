clear all; close all; clc;

% Script for reproducing the experiments on Euclidean Distance Matrix
% Completion described in
% 
% Quartic First-Order Methods for Low Rank Minimization,
% arXiv preprint arXiv:1901.10791
%
% Author
% Radu-Alexandru Dragomir [radu-alexandru.dragomir@inria.fr]

% ======== GENERATING THE PROBLEM =======


%problem dimensions
n = 5000;
r = 3; 

% number of runs to average
n_runs = 1;

% Fraction of unknown distances
fractionOfUnknown = 0.95;
p = 1 - fractionOfUnknown;
    
% generating the Helix data
To = 2 * pi * rand(n,1);
Yo = zeros(n,r);
Yo(:,1) = cos(To * 3.);
Yo(:,2) = sin(To * 3.);
Yo(:,3) = 2 * To;

Yo = Yo / norm(Yo, 'fro');

trueDists = pdist(Yo)'.^2; % true distances

test = false(length(trueDists),1);
test(1:floor(length(trueDists)*fractionOfUnknown)) = true;
test = test(randperm(length(test)));
train = ~test;
m = sum(train);


% ======= ALGORITHM PARAMETERS =====

params.maxiter = 1000;
params.pmax = r;
params.tol = 1e-28;
params.vtol = 1e-28;
params.verb = true;

params.gram_inner_tol = 1e-7;
params.gram_max_iter = 50;
params.max_step = 1;
params.ls_maxiter = 20;

params.maxiter_tr = 50;
params.monitor_rmse = true;
params.monitor_interval = 20;

% Compute all pair of indices
H = tril(true(n),-1);
[I,J] = ind2sub([n,n],find(H(:))); 
clear 'H';

% setting parameters for NoLips
% in this implementation, the objective function is divided by m
% hence we scale the constants accordingly
quartic_param = 9 * n * p / m;
quad_param    = 2 * norm(trueDists(train)) / m;
params.alpha = 2*quartic_param;
params.beta  = quartic_param;
params.sigma = quad_param;

%% ==== DOING SEVERAL RUNS =====
time_values = cell(1,4); 
rmse_values = cell(1,4);
all_infos = cell(n_runs,4);

for run = 1:n_runs

    % Initial condition$
    Y0 = randn(n,r);
    m = sum(train);
    
    % Run algorithms
    % norm kernel
    params.alpha = 6 * quartic_param;
    params.beta = 0;
    [Y_bg, infos_bg] = bg_dist_completion(I(train),J(train),trueDists(train),Y0,trueDists,params);
    all_infos{run,1} = infos_bg;
    
    % Gram kernel
    params.beta  = quartic_param;
    params.alpha = quartic_param;

    [Y_bg2, infos_bg2] = bg_dist_completion(I(train),J(train),trueDists(train),Y0,trueDists,params);
    all_infos{run,2} = infos_bg2;

    % Gradient descent
    [Y_gd, infos_gd] = gd_dist_completion(I(train),J(train),trueDists(train),Y0,trueDists,params);
    all_infos{run,3} = infos_gd;
    
    % Riemannian Trust-region algorithm
    params.tol = 1e-15;

    [Y_tr, infos_tr] = tr_dist_completion(I(train),J(train),trueDists(train),Y0,trueDists,params);
    all_infos{run,4} = infos_tr;
end


%%====== Averaging values of the different runs ========

time_values = cell(1,4);
rmse_values = cell(1,4);

for algo = 1:4
    time_values{algo} = all_infos{1,algo}.rmse_time;
    rmse_values{algo} = all_infos{1,algo}.rmse;
    
    for run = 2:n_runs
        n_meas_old = size(time_values{algo},1);
        n_meas_new = size(all_infos{run,algo}.rmse_time,1);
        n_meas_new = min(n_meas_old, n_meas_new);
        
        time_values{algo} = time_values{algo}(1:n_meas_new,:) + all_infos{run,algo}.rmse_time(1:n_meas_new);
        rmse_values{algo} = rmse_values{algo}(1:n_meas_new,:) + all_infos{run,algo}.rmse(1:n_meas_new);
    end
    
    time_values{algo} = time_values{algo} / n_runs;
    rmse_values{algo} = rmse_values{algo} / n_runs;
end

%% PLOTTING

% plot properties
width = 4;     % Width in inches
height = 4;    % Height in inches
alw = 1;    % AxesLineWidth
fsz = 12;      % Fontsize
lw = 1.8;      % LineWidth
msz = 8;       % MarkerSize

pos = get(gcf, 'Position');

linestyles = {'r','-.','--','-x'};

for algo = 1:4
    semilogy(time_values{algo}, rmse_values{algo},linestyles{algo},'LineWidth',lw,'MarkerSize',msz);
    hold on

end

legend('Dyn-NoLips', 'Dyn-NoLips-Gram','GD + Armijo', 'Trust region');

xlabel('CPU Time (s)');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

ylabel('RMSE');
