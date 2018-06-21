clear all; close all; clc;
set(0, 'DefaultFigureWindowStyle', 'normal');
currentFolder = pwd;
addpath(genpath(pwd));

% Include dependencies
addpath('./FSLib/lib'); % dependencies
addpath('./FSLib/methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% featsRange = 12;
featsRange = [6, 12, 20];   % select the first 2 features
numIterations = 10;           % number of iterations for the loop

% Select a feature selection method from the list
listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fscm','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs', 'ofs', 'pdfadfs'};
[methodID] = readInput(listFS);
selection_method = listFS{methodID}; % Selected

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the data
% Holdout

x = load('inputs (not separated).mat');
y = load('targets (not separated).mat');
X = x.netTrainInputs;
Y = y.netTrainTargets;
Y = rescale(Y, -1, 1);

CVO = cvpartition(Y,'Holdout',0.20);
X_train = double(X(CVO.training,:));
Y_train = double(Y(CVO.training));
X_test = double(X(CVO.test,:));
Y_test = double(Y(CVO.test));

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
        options.k = 5; %For unsupervised feature selection, you should tune
        %this parameter k, the default k is 5.
        options.nUseEigenfunction = 4;  %You should tune this parameter.
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
        ranking = UDFS(X_train , nClass); 
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
r = ranking';

for numFeats = featsRange
    cath_error_arr = zeros(numIterations, 1);
    cath_accuracy_arr = zeros(numIterations, 1);
    confusionMatrix = cell(numIterations,1);
    tic;
    for i = 1:numIterations
        rng(i);
        svmClassifier = fitcsvm(X_train(:,ranking(1:numFeats)), ...
                            Y_train, ...
                            'KernelFunction', 'linear', ...
                            'Standardize', true, ...
                            'Verbose', 0);
        [C, scores] = predict(svmClassifier, X_test(:,ranking(1:numFeats)));
        conMat = confusionmat(Y_test,C);
        cath_accuracy = conMat(2,2)/(conMat(2,1) + conMat(2,2));
        cath_error = 1 - cath_accuracy;
        cath_accuracy_arr(i,1) = cath_accuracy; 
        cath_error_arr(i,1) = cath_error;
        confusionMatrix{i,1} = conMat;
    end
    oneIterTime = toc/numIterations;
    meanAccuracy = mean(cath_accuracy_arr);
    meanSTD = round(std(cath_accuracy_arr));
    fprintf('\nMethod %s for %d features (Linear-SVMs): Accuracy: %.2f±%.2f%%, Error-Rate: %.2f%%\n', ...
            selection_method, numFeats, 100*meanAccuracy, 100*meanSTD, 100*mean(cath_error));
    fprintf('Elapsed time: %.2f\n', oneIterTime);
    % Save data
    methodName = strcat(selection_method, '_', num2str(numFeats), '_features.mat'); 
    cd('FS methods comparison');
    save(methodName, 'cath_accuracy_arr', 'cath_error_arr', 'confusionMatrix', 'numIterations', 'oneIterTime');
    cd ..\
end