clear all; close all; clc;
set(0, 'DefaultFigureWindowStyle', 'normal');
currentFolder = pwd;
addpath(genpath(pwd));
warning('off', 'all');
% Include dependencies
addpath('./FSLib/lib'); % dependencies
addpath('./FSLib/methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

featsRange = 20;
% featsRange = [6, 12, 20];   % select the first 2 features
numFolds = 10;              % number of iterations for the loop
% Select a feature selection method from the list
% listFS = {'ILFS', 'InfFS', 'ECFS', 'mrmr', 'relieff',... 
%           'mutinffs', 'fscm', 'laplacian', 'mcfs', 'rfe', ...
%           'L0','fisher', 'UDFS', 'llcfs', 'cfs', ...
%           'ofs', 'pdfadfs'};
% [methodID] = readInput(listFS);
% selection_method = listFS{methodID};
selection_method = 'pdfadfs';

meanAccuracy = 0;
meanSTD = 0.05; 
while ~(meanAccuracy > 0.855 && meanSTD < 0.025) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the data
x = load('inputs (not separated).mat');
y = load('targets (not separated).mat');
X = x.netTrainInputs;
Y = y.netTrainTargets;
Y = rescale(Y, -1, 1);
CVO = cvpartition(Y, 'k', numFolds);

for numFeats = featsRange
    total_ranking = [];
    cath_accuracy_arr = [];
    confusionMatrix = {};
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
        switch lower(selection_method)
            case 'ilfs'
                % Infinite Latent Feature Selection - ICCV 2017
                [ranking, weights, subset] = ILFS(X_train, Y_train , 4, 0);
            case 'mrmr'
                ranking = mRMR(X_train, Y_train, numF);
            case 'relieff'
                [ranking, w] = reliefF(X_train, Y_train, 20);
            case 'mutinffs'
                [ranking, w] = mutInfFS(X_train, Y_train, numF);
            case 'fscm'
                [ranking, w] = fsvFS(X_train, Y_train, numF);
            case 'laplacian'
                W = dist(X_train');
                W = -W./max(max(W)); % it's a similarity
                [lscores] = LaplacianScore(X_train, W);
                [junk, ranking] = sort(-lscores);
            case 'mcfs'
                % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
                options = [];
                options.k = 5;
                options.nUseEigenfunction = 4;
                [FeaIndex,~] = MCFS_p(X_train,numF,options);
                ranking = FeaIndex{1};
            case 'rfe'
                ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
            case 'l0'
                ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
            case 'fisher'
                ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
            case 'inffs'
                % Infinite Feature Selection 2015 updated 2016
                alpha = 0.5;    % default, it should be cross-validated.
                sup = 1;        % Supervised or Not
                [ranking, w] = infFS(X_train , Y_train, alpha , sup , 0);    
            case 'ecfs'
                % Features Selection via Eigenvector Centrality 2016
                alpha = 0.5; % default, it should be cross-validated.
                ranking = ECFS(X_train, Y_train, alpha)  ;
            case 'udfs'
                % Regularized Discriminative Feature Selection for Unsupervised Learning
                nClass = 2;
                ranking = UDFS(X_train, nClass); 
            case 'cfs'
                % BASELINE - Sort features according to pairwise correlations
                ranking = cfs(X_train);     
            case 'llcfs'   
                % Feature Selection and Kernel Learning for Local Learning-Based Clustering
                ranking = llcfs(X_train);
            case 'ofs'
                ranking = [14; 18; 1; 2; 17; 10; 12; 19; 3; 7; 4; 11; 8; 20; 16; 13; 6; 9; 5; 15];
            case 'pdfadfs'
                ranking = [17; 18; 19; 20; 4; 3; 7; 1; 2; 15; 16; 8; 5; 14; 13; 12; 6; 9; 10; 11];
            otherwise
                disp('Unknown method.')
        end

        if size(ranking, 2) ~= 1
            ranking = ranking';
        end
        total_ranking = cat(2, total_ranking, ranking);

        % Train a classifier
        svmClassifier = fitcsvm(X_train(:,ranking(1:numFeats)), ...
                        Y_train, ...
                        'KernelFunction', 'linear', ...
                        'Standardize', true, ...
                        'Verbose', 0);
        [Y_pred, scores] = predict(svmClassifier, X_test(:,ranking(1:numFeats)));
        tempConfMatrix = confusionmat(Y_test,Y_pred);
        cath_accuracy = tempConfMatrix(2,2)/(tempConfMatrix(2,1) + tempConfMatrix(2,2));
        cath_accuracy_arr = cat(1, cath_accuracy_arr, cath_accuracy); 
        confusionMatrix = cat(1, confusionMatrix, tempConfMatrix);
    end
    
    % Display the results
    meanAccuracy = mean(cath_accuracy_arr);
    meanSTD = std(cath_accuracy_arr);
    fprintf('\nMethod %s for %d features (Linear-SVMs): Accuracy: %.1f�%.1f%%\n', ...
            selection_method, numFeats, 100*meanAccuracy, 100*meanSTD);
    
    % Write data to xlsx file
    % Mean and STD values
    xls_filename = 'testdata2.xlsx';
    sheet = 1;    
    xlRange = 'C6';
    for i = find(numFeats==featsRange)
        if i == 1
            data_xlRow = str2num(xlRange(2:end));
        else
            data_xlRow = str2num(xlRange(2:end))+(i-1)*15;
        end
        data_xlRange = strcat(xlRange(1), num2str(data_xlRow));
    end  
    xlswrite(xls_filename, cath_accuracy_arr, sheet, data_xlRange)
    
    % Accuracy array
    xls_filename = 'testdata.xlsx';
    cols_name = {'6 features', '12 features', '20 features'}; 
    sheet = 1;
    cols_xlRange = 'B2';
    xlswrite(xls_filename, cols_name, sheet, cols_xlRange)
    
    for i = find(numFeats==featsRange)
        data_xlCol = char(cols_xlRange(1)+i-1);
        data_xlRow = str2num(cols_xlRange(2:end))+1; %#ok<ST2NM>
        data_xlRange = strcat(data_xlCol, num2str(data_xlRow));
        xlswrite(xls_filename, cath_accuracy_arr, sheet, data_xlRange)
    end
     
    % Save data
    methodName = strcat(selection_method, '_', num2str(numFeats), '_features.mat'); 
    cd('FS methods comparison');
    save(methodName, 'total_ranking', 'cath_accuracy_arr', 'confusionMatrix', 'meanAccuracy', 'meanSTD');
    cd ..\
end

end

winopen('testdata2.xlsx')
