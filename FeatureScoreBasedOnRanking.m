clc; clear;
set(0, 'DefaultFigureWindowStyle', 'normal');
currentFolder = pwd;
addpath(genpath(pwd));

trainingType = 'svm_uci'; % svm or svm_uci
if strcmp(trainingType, 'svm') 
    sheet = 'FS comparison (SVM)';
    xlRange = 'C67:P86';
elseif strcmp(trainingType, 'svm_uci') 
    sheet = 'FS comparison (SVM UCI)';
    xlRange = 'C67:P96';
end

rankingTable = xlsread('Feature engineering (not separated).xlsm', sheet, xlRange);
nFeatures = size(rankingTable, 1);
nRows = size(rankingTable, 1);
scores = zeros(nFeatures, 1);

for feature = 1:nFeatures
    score = 0;
    for row = 1:nRows
        count = sum(squeeze(rankingTable(row, :)) == feature);
        score = score + count * (nRows - row + 1);
    end
    scores(feature) = score;
end
