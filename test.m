procTime = [];

for i = 1:19
    tic;
    distBhatt(1, i) = GetBhattacharyyaDistance(X_train(:, i), X_train(:, i+1)); 
    tempProcTime = toc;
    procTime = cat(1, procTime, tempProcTime);
end

