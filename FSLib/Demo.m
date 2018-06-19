clear all; close all; clc
% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

k = 2; % select the first 2 features
isLoadSeparatedData = 0;    % separated data = 1, not separated data = 0
useNormalizedData = 0;      % 1 - yes, 0 - no (as IP said default value is euqal to 0)

% Select a feature selection method from the list
listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};
[methodID] = readInput( listFS );
selection_method = listFS{methodID}; % Selected

% % Load the data
% if isLoadSeparatedData == 0
%     x = load('inputs (not separated).mat');
%     y = load('targets (not separated).mat');
% else
%     x = load('inputs (separated).mat');
%     y = load('targets (separated).mat');
% end
% 
% if useNormalizedData == 1
%     x = x.netTrainInputsNorm;
%     y = y.netTrainTargetsNorm;
% elseif useNormalizedData == 0
%     X = x.netTrainInputs;
%     Y = y.netTrainTargets;
% end
% 
% P = cvpartition(Y,'Holdout',0.20);
% X_train = double( X(P.training,:) );
% Y_train = (double( Y(P.training) )-1)*2-1; % labels: neg_class -1, pos_class +1
% 
% X_test = double( X(P.test,:) );
% Y_test = (double( Y(P.test) )-1)*2-1; % labels: neg_class -1, pos_class +1
% 
% % number of features
% numF = size(X_train,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the data and select features for classification
load fisheriris
X = meas; clear meas
% Extract the Setosa class
Y = nominal(ismember(species,'setosa')); clear species

% Randomly partitions observations into a training set and a test
% set using stratified holdout
P = cvpartition(Y,'Holdout',0.20);

X_train = double( X(P.training,:) );
Y_train = (double( Y(P.training) )-1)*2-1; % labels: neg_class -1, pos_class +1

X_test = double( X(P.test,:) );
Y_test = (double( Y(P.test) )-1)*2-1; % labels: neg_class -1, pos_class +1

% number of features
numF = size(X_train,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% feature Selection on training data
switch lower(selection_method)
    case 'ilfs'
        % Infinite Latent Feature Selection - ICCV 2017
        [ranking, weights, subset] = ILFS(X_train, Y_train , 4, 0 );
    case 'mrmr'
        ranking = mRMR(X_train, Y_train, numF);
        
    case 'relieff'
        [ranking, w] = reliefF( X_train, Y_train, 20);
        
    case 'mutinffs'
        [ ranking , w] = mutInfFS( X_train, Y_train, numF );
        
    case 'fsv'
        [ ranking , w] = fsvFS( X_train, Y_train, numF );
        
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
        [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );    
        
    case 'ecfs'
        % Features Selection via Eigenvector Centrality 2016
        alpha = 0.5; % default, it should be cross-validated.
        ranking = ECFS( X_train, Y_train, alpha )  ;
        
    case 'udfs'
        % Regularized Discriminative Feature Selection for Unsupervised Learning
        nClass = 2;
        ranking = UDFS(X_train , nClass ); 
        
    case 'cfs'
        % BASELINE - Sort features according to pairwise correlations
        ranking = cfs(X_train);     
        
    case 'llcfs'   
        % Feature Selection and Kernel Learning for Local Learning-Based Clustering
        ranking = llcfs( X_train );
        
    otherwise
        disp('Unknown method.')
end

% Use a linear support vector machine classifier
svmStruct = fitcsvm(X_train(:,ranking(1:k)),Y_train);
C = predict(svmStruct,X_test(:,ranking(1:k)));
err_rate = sum(Y_test~= C)/P.TestSize; % mis-classification rate
conMat = confusionmat(Y_test,C); % the confusion matrix

disp('X_train size')
size(X_train)

disp('Y_train size')
size(Y_train)

disp('X_test size')
size(X_test)

disp('Y_test size')
size(Y_test)


fprintf('\nMethod %s (Linear-SVMs): Accuracy: %.2f%%, Error-Rate: %.2f%% \n', selection_method, 100*(1-err_rate),100*err_rate);

