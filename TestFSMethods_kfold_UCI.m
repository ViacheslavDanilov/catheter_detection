clear all; close all; clc;
set(0, 'DefaultFigureWindowStyle', 'normal');
currentFolder = pwd;
addpath(genpath(pwd));
warning('off', 'all');
% Include dependencies
addpath('./FSLib/lib'); % dependencies
addpath('./FSLib/methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% featsRange = 30;                   % Only for 1 feature
featsRange = 1:1:30;              % For all feature range
numFolds = 10;              % number of iterations for the loop
isGetRanking = 1;           % Do not perform calculations (if 0) and using already defined rankings
isSave = 0;                 % Save main data variables

% Select a feature selection method from the list
listFS = {'ILFS', 'INFS', 'ECFS', 'MRMR', 'RFFS', ... 
          'MIFS', 'FSCM', 'LSFS', 'MCFS', 'UDFS', ... 
          'CFS', 'BDFS', 'OFS', 'ADFS'};
[methodID] = readInput(listFS);
selection_method = listFS{methodID};

% Load the data
dataBreastStruct = load('breastData.mat');
X = dataBreastStruct.dataBreast;
Y = dataBreastStruct.dataLabels;
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
        if isGetRanking == 1 && i == 1
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
                    ranking = [24;8;13;28;11;4;23;21;3;1;7;27;14;26;6;29;22;25;18;2;30;5;9;16;20;19;17;15;12;10];
                case 'OFS'
                    ranking = [17;14;20;13;11;15;19;12;16;30;29;5;9;10;18;2;25;24;26;22;6;1;4;3;27;23;7;21;8;28];
                case 'ADFS'
                    ranking = [28;8;7;27;4;24;3;1;21;23;26;6;18;22;25;2;11;5;30;16;13;9;29;14;19;10;15;17;12;20];  
                otherwise
                    disp('Unknown method.')
            end
        elseif isGetRanking == 0 && i == 1
            calcFilename = 'Feature engineering (not separated).xlsm';
            sheetName = 'FS comparison (SVM_UCI)';
            switch selection_method
                case 'ILFS'
                    rankingRange = 'C67:C96';
                case 'INFS'
                    rankingRange = 'D67:D96';
                case 'ECFS'
                    rankingRange = 'E67:E96';
                case 'MRMR'
                    rankingRange = 'F67:F96';
                case 'RFFS'
                    rankingRange = 'G67:G96';
                case 'MIFS'
                    rankingRange = 'H67:H96';
                case 'FSCM'
                    rankingRange = 'I67:I96';
                case 'LSFS'
                    rankingRange = 'J67:J96';
                case 'MCFS'
                    rankingRange = 'K67:K96';             
                case 'UDFS'
                    rankingRange = 'L67:L96';
                case 'CFS'
                    rankingRange = 'M67:M96';
                case 'BDFS'
                    rankingRange = 'N67:N96';
                case 'OFS'
                    rankingRange = 'O67:O96';
                case 'ADFS'
                    rankingRange = 'P67:P96';
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
        % ranking  = [18;4;14;17;15;6;10;9;12;8;13;16;11;19;5;7;3;20;1;2;];

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
    
    if sum(average_ranking) ~= 465
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
xls_filename = 'svmMultipleTestUCI.xlsx';
sheet = 1;
means_xlRange = 'B2';
stds_xlRange = 'B35';
time_xlRange = 'B68';
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

winopen('svmMultipleTestUCI.xlsx')




