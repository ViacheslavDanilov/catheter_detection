%% Estimate scores of PDF distribution (ADFS)
clc; clear;
addpath('MAT files');
load('data (not separ.).mat');
load('breastData.mat');
binCounts = 10:2:70;
% binCounts = 10;

dataCatheter = dataBreast1;
dataTissue = dataBreast2;

featureCount = size(dataCatheter,2);
scoresADFS = zeros(numel(binCounts), featureCount);
for i = 1:numel(binCounts)
    scoresADFS(i,1) = binCounts(i);
    for j = 2:featureCount % test all features
%     for j = 10  % Test only one feautre (use ID of a feature)
        isDiscrete = IsDiscrete(dataCatheter(:,j));
        scoresADFS(i,j) = ShowPDFDifference(dataTissue(:,j), dataCatheter(:,j), binCounts(i), binCounts(i), isDiscrete);
%         scoresADFS(i,j) = ShowPDFDifference(dataTissue(:,j), dataCatheter(:,j), binCounts(i), binCounts(i), isDiscrete, 'sPlot');
        close
    end
end

%% Estimate scores of overlapping (OFS)
epsilonCounts = 0.01:0.01:0.10;
scoresOFS = zeros(1, featureCount);
for i = 1:numel(epsilonCounts)
    scoresOFS(i,1) = epsilonCounts(i);
    for j = 2:featureCount
        scoresOFS(i,j) = GetOverlapRate(dataTissue(:,j), dataCatheter(:,j), epsilonCounts(i));
    end
end

%% Estimate scores of overlapping (OFS)
scoresBDFS = zeros(1, featureCount);
for j = 2:featureCount
    scoresBDFS(1,j) = GetBhattacharyyaDistance(dataTissue(:,j), dataCatheter(:,j));
end

