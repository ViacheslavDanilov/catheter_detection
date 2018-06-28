clear all; close all; clc;
set(0, 'DefaultFigureWindowStyle', 'normal');
currentFolder = pwd;
addpath(genpath(pwd));
warning('off', 'all');
% Include dependencies
addpath('./FSLib/lib'); % dependencies
addpath('./FSLib/methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% featsRange = 4;                   % Only for 1 feature
% featsRange = [3, 6, 12, 20];      % Only for 4 features
featsRange = 1:1:20;
numFolds = 10;              % number of iterations for the loop
isGetRanking = 0;           % Do not perform calculations (if 0) and using already defined rankings
isSave = 0;                 % Save main data variables
isWriteToXLS = 0;           % Write data to xls file
% Select a feature selection method from the list
listFS = {'ILFS', 'INFS', 'ECFS', 'MRMR', 'RFFS', ... 
          'MIFS', 'FSCM', 'LSFS', 'MCFS', 'UDFS', ... 
          'CFS', 'BDFS', 'OFS', 'PDF ADFS'};
[methodID] = readInput(listFS);
selection_method = listFS{methodID};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEMPORAL TESTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% selection_method = 'PDF ADFS';
% meanAccuracy = 0;
% meanSTD = 0.005; 
% % while ~(meanAccuracy < 0.82 && meanSTD > 0.04) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
                case 'udfs'
                    nClass = 2;
                    ranking = UDFS(X_train, nClass); 
                case 'CFS'
                    ranking = cfs(X_train);     
                case 'BDFS'
                    ranking = [2; 1; 3; 20; 19; 7; 18; 5; 13; 15; 6; 12; 16; 17; 9; 14; 8; 4; 11; 10];
                case 'OFS'
                    ranking = [14; 18; 1; 2; 17; 10; 12; 19; 3; 7; 4; 11; 8; 20; 16; 13; 6; 9; 5; 15];
                case 'PDF ADFS'
                    ranking = [17; 18; 19; 20; 4; 3; 7; 1; 2; 15; 16; 8; 5; 14; 13; 12; 6; 9; 10; 11];  
                otherwise
                    disp('Unknown method.')
            end
        else
            calcFilename = 'Feature engineering (not separated).xlsm';
            sheetName = 'FS comparison (SVM) (1)';
            switch selection_method
                case 'ILFS'
                    rankingRange = 'C82:C101';
                case 'INFS'
                    rankingRange = 'D82:D101';
                case 'ECFS'
                    rankingRange = 'E82:E101';
                case 'MRMR'
                    rankingRange = 'F82:F101';
                case 'RFFS'
                    rankingRange = 'G82:G101';
                case 'MIFS'
                    rankingRange = 'H82:H101';
                case 'FSCM'
                    rankingRange = 'I82:I101';
                case 'LSFS'
                    rankingRange = 'J82:J101';
                case 'MCFS'
                    rankingRange = 'K82:K101';             
                case 'UDFS'
                    rankingRange = 'L82:L101';
                case 'CFS'
                    rankingRange = 'M82:M101';
                case 'BDFS'
                    rankingRange = 'N82:N101';
                case 'OFS'
                    rankingRange = 'O82:O101';
                case 'PDF ADFS'
                    rankingRange = 'P82:P101';
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
    
    if sum(average_ranking) ~= 210
%         f = msgbox('Possible problems with ranking!', 'Warning');
        fprintf('Possible problems with ranking!');
    end
    
    % Display the results
    meanAccuracy = mean(cath_accuracy_arr);
    meanSTD = std(cath_accuracy_arr);
    fprintf('\nMethod %s for %d features (Linear-SVMs): accuracy = %.1f�%.1f%%, ranking time = %.2f, training time = %.2f\n', ...
            selection_method, numFeats, 100*meanAccuracy, 100*meanSTD, mean(rankingTime), mean(trainingTime));
    
    % Write data to xlsx file
    % Mean and STD values
    if isWriteToXLS == 1
        xls_filename = 'testdata2.xlsx';
        sheet = 1;    
        xlRange = 'B2';
        xlRange_time = 'C2';
        xlRange_ranking = 'D2';
        for i = find(numFeats==featsRange)
            if i == 1
                data_xlRow = str2num(xlRange(2:end));
                time_xlRow = str2num(xlRange_time(2:end));
            else
                data_xlRow = str2num(xlRange(2:end))+(i-1)*15;
                time_xlRow = str2num(xlRange_time(2:end))+(i-1)*10;
            end
            data_xlRange = strcat(xlRange(1), num2str(data_xlRow));
            time_xlRange = strcat(xlRange_time(1), num2str(time_xlRow));
        end  
        xlswrite(xls_filename, cath_accuracy_arr, sheet, data_xlRange)
        xlswrite(xls_filename, rankingTime, sheet, time_xlRange)
        xlswrite(xls_filename, average_ranking, sheet, xlRange_ranking)
    end
    
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
xls_filename = 'testdata.xlsx';
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

winopen('testdata.xlsx')




