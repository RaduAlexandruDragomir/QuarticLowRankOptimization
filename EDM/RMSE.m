function output = RMSE(Y,trueDists)
% Returns the normalized root mean squared error
% between the pairwise distances between rows of Y and the ground truth distances
% trueDists
    
    dists = pdist(Y)';
    mse = mean((sqrt(trueDists) - dists) .^ 2) / mean(trueDists);
    output = sqrt(mse);
end

