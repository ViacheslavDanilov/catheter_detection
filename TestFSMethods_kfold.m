clear all; close all; clc;
set(0, 'DefaultFigureWindowStyle', 'normal');
currentFolder = pwd;
addpath(genpath(pwd));
warning('off', 'all');
% Include dependencies
addpath('./FSLib/lib'); % dependencies
addpath('./FSLib/methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% CHECK STRING 144(used for manual rankings)
% featsRange = 12;                   % Only for 1 feature
% featsRange = [3, 6, 12, 20];      % Only for 4 features
featsRange = 1:1:20;

numFolds = 10;              % number of iterations for the loop
isGetRanking = 0;           % Do not perform calculations (if 0) and using already defined rankings
isSave = 0;                 % Save main data variables

% Select a feature selection method from the list
listFS = {'ILFS', 'INFS', 'ECFS', 'MRMR', 'RFFS', ... 
          'MIFS', 'FSCM', 'LSFS', 'MCFS', 'UDFS', ... 
          'CFS', 'BDFS', 'OFS', 'ADFS'};
[methodID] = readInput(listFS);
selection_method = listFS{methodID};

% Load the data
x = load('inputs (not separated).mat');
y = load('targets (not separated).mat');
X = x.netTrainInputs;
Y = y.netTrainTargets;
Y = rescale(Y, -1, 1);
CVO = cvpartition(Y, 'k', numFolds);

svmOutput = {};
for numFeats = featsRange   
    total_ranking = [];
    cath_accuracy_arr = [];
    confusionMatrix = {};
    rankingTime = [];
    trainingTime = [];
    for i = 1:CVO.NumTestSets   
        trainIdx = CVO.training(i);
        testIdx = CVO.test(i);
        X_train = X(trainIdx,:);
        Y_train = Y(trainIdx,:);
        X_test = X(testIdx,:);
        Y_test = Y(testIdx,:);

        % Number of features
        numF = size(X_train,2);

        % Feature Selection on training data
        tic;
        if isGetRanking == 1
            switch selection_method
                case 'ILFS'
                    [ranking, weights, subset] = ILFS(X_train, Y_train , 4, 0);
                case 'INFS'
                    alpha = 0.5;
                    sup = 1;
                    [ranking, w] = infFS(X_train , Y_train, alpha , sup , 0);
                case 'ECFS'
                    alpha = 0.5;
                    ranking = ECFS(X_train, Y_train, alpha);
                case 'MRMR'
                    ranking = mRMR(X_train, Y_train, numF);
                case 'RFFS'
                    [ranking, w] = reliefF(X_train, Y_train, 20);
                case 'MIFS'
                    [ranking, w] = mutInfFS(X_train, Y_train, numF);
                case 'FSCM'
                    [ranking, w] = fsvFS(X_train, Y_train, numF);
                case 'LSFS'
                    W = dist(X_train');
                    W = -W./max(max(W));
                    [lscores] = LaplacianScore(X_train, W);
                    [junk, ranking] = sort(-lscores);
                case 'MCFS'
                    options = [];
                    options.k = 5;
                    options.nUseEigenfunction = 4;
                    [FeaIndex,~] = MCFS_p(X_train,numF,options);
                    ranking = FeaIndex{1};
                case 'UDFS'
                    nClass = 2;
                    ranking = UDFS(X_train, nClass); 
                case 'CFS'
                    ranking = cfs(X_train);     
                case 'BDFS'
                    ranking = [2; 1; 3; 20; 19; 7; 18; 5; 13; 15; 6; 12; 16; 17; 9; 14; 8; 4; 11; 10];
                case 'OFS'
                    ranking = [14; 18; 1; 2; 17; 10; 12; 19; 3; 7; 4; 11; 8; 20; 16; 13; 6; 9; 5; 15];
                case 'ADFS'
                    ranking = [17; 18; 19; 20; 4; 3; 7; 1; 2; 15; 16; 8; 5; 14; 13; 12; 6; 9; 10; 11];  
                otherwise
                    disp('Unknown method.')
            end
        else
            calcFilename = 'Feature engineering (not separated).xlsm';
            sheetName = 'FS comparison (SVM)';
            switch selection_method
                case 'ILFS'
                    rankingRange = 'C67:C86';
                case 'INFS'
                    rankingRange = 'D67:D86';
                case 'ECFS'
                    rankingRange = 'E67:E86';
                case 'MRMR'
                    rankingRange = 'F67:F86';
                case 'RFFS'
                    rankingRange = 'G67:G86';
                case 'MIFS'
                    rankingRange = 'H67:H86';
                case 'FSCM'
                    rankingRange = 'I67:I86';
                case 'LSFS'
                    rankingRange = 'J67:J86';
                case 'MCFS'
                    rankingRange = 'K67:K86';             
                case 'UDFS'
                    rankingRange = 'L67:L86';
                case 'CFS'
                    rankingRange = 'M67:M86';
                case 'BDFS'
                    rankingRange = 'N67:N86';
                case 'OFS'
                    rankingRange = 'O67:O86';
                case 'ADFS'
                    rankingRange = 'P67:P86';
                otherwise
                    disp('Unknown method.')
            end
            ranking = xlsread(calcFilename, sheetName, rankingRange);
        end    
        tempRankingTime = toc;
        rankingTime = cat(1, rankingTime, tempRankingTime);
        
        if size(ranking, 2) ~= 1
            ranking = ranking';
        end
        total_ranking = cat(2, total_ranking, ranking);
        
        % Manual ranking
        % ranking  = [18;17;4;19;14;15;7;5;3;20;2;1;10;16;12;6;13;9;8;11];

        % Train a classifier
        tic;
        svmClassifier = fitcsvm(X_train(:,ranking(1:numFeats)), ...
                        Y_train, ...
                        'KernelFunction', 'linear', ...
                        'Standardize', true, ...
                        'Verbose', 0);
        tempTrainingTime = toc;
        trainingTime = cat(1, trainingTime, tempTrainingTime); 
        [Y_pred, scores] = predict(svmClassifier, X_test(:,ranking(1:numFeats)));
        tempConfMatrix = confusionmat(Y_test, Y_pred);
        cath_accuracy = tempConfMatrix(2,2)/(tempConfMatrix(2,1) + tempConfMatrix(2,2));
        cath_accuracy_arr = cat(1, cath_accuracy_arr, cath_accuracy); 
        confusionMatrix = cat(1, confusionMatrix, tempConfMatrix);
    end
    
    % Writing all results into one variable
    svmOutput{numFeats, 1} = cath_accuracy_arr;
    svmOutput{numFeats, 2} = confusionMatrix;
    svmOutput{numFeats, 3} = rankingTime;
    svmOutput{numFeats, 4} = trainingTime;
    svmOutput{numFeats, 5} = total_ranking';
    
    % Find average ranking
    average_ranking = zeros(size(total_ranking,1), 1);    
    for k = 1:size(total_ranking,1)
        tempArray = total_ranking(k,:);
        mostFreqVal = mode(tempArray, 2);
        average_ranking(k,1) = mostFreqVal;
    end
    
    if sum(average_ranking) ~= 210
        fprintf('Possible problems with ranking!');
    end
    
    % Display the results
    meanAccuracy = mean(cath_accuracy_arr);
    meanSTD = std(cath_accuracy_arr);
    fprintf('\nMethod %s for %d features (Linear-SVMs): accuracy = %.1f±%.1f%%, ranking time = %.2f, training time = %.2f\n', ...
            selection_method, numFeats, 100*meanAccuracy, 100*meanSTD, mean(rankingTime), mean(trainingTime));
    
    % Save data
    if isSave == 1
        methodName = strcat(selection_method, '_', num2str(numFeats), '_features.mat'); 
        cd('FS methods comparison/Separated outputs');
        save(methodName, 'total_ranking', 'cath_accuracy_arr', 'pTime', ...
            'confusionMatrix', 'meanAccuracy', 'meanSTD', 'average_ranking');
        cd ..\..
    end
end
% end

% Save SVM output
svmOutputName = strcat(selection_method, '_', num2str(numFeats), '_features.mat'); 
cd('FS methods comparison');
save(svmOutputName, 'svmOutput');
cd ..\

% Save means and STDs for all features
xls_filename = 'svmMultipleTest.xlsx';
sheet = 1;
means_xlRange = 'B2';
stds_xlRange = 'B25';
time_xlRange = 'B48';
featureMeans = zeros(numF, 1);
featureSTDs = zeros(numF, 1);
featureTrainingTime = zeros(numF, 1);
for i = 1:size(svmOutput, 1)
    featureMeans(i,1) = mean(svmOutput{i,1});
    featureSTDs(i,1) = std(svmOutput{i,1});
    featureTrainingTime(i,1) = mean(svmOutput{i,4});
end       
xlswrite(xls_filename, featureMeans, sheet, means_xlRange);
xlswrite(xls_filename, featureSTDs, sheet, stds_xlRange);
xlswrite(xls_filename, featureTrainingTime, sheet, time_xlRange);

winopen('svmMultipleTest.xlsx')




