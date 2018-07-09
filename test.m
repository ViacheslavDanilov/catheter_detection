%% Estimate scores of overlapping (OFS)
scoresBDFS = zeros(1, featureCount);
for j = 2:featureCount
    scoresBDFS(1,j) = GetBhattacharyyaDistance(dataTissue(:,j), dataCatheter(:,j));
    disp(j);
end
