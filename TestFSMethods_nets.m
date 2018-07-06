clear all; close all; clc;
set(0, 'DefaultFigureWindowStyle', 'normal');
currentFolder = pwd;
addpath(genpath(pwd));
warning('off', 'all');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initial Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
featsRange = 1:1:20; % could be a range or a digit
numIterations = 10;
isUseGPU = 1;
netType = 'cascade';   % 'feed-forward', 'cascade', 'recurrent'
netSize = 'small';
isSave = 1;
isVisual = 0;

if isUseGPU == 1
    trainingFunction = 'SCG';
    perfFunction = 'crossentropy';
else
    trainingFunction = 'BR';
    perfFunction = 'mse';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
listFS = {'ILFS', 'INFS', 'ECFS', 'MRMR', 'RFFS', ... 
          'MIFS', 'FSCM', 'LSFS', 'MCFS', 'UDFS', ... 
          'CFS', 'BDFS', 'OFS', 'ADFS'};
[methodID] = readInput(listFS);
selection_method = listFS{methodID};

% Get ranking
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
    case 'ADFS'
        rankingRange = 'P82:P101';
    otherwise
        disp('Unknown method.')
end

% Load the data
dataFile = 'Feature engineering (not separated).xlsm';
sheet = 'FS comparison (SVM)';
ranking = xlsread(dataFile, sheet, rankingRange);    
x = load('inputs (not separated).mat');
y = load('targets (not separated).mat');
X = x.netTrainInputs;
Y = y.netTrainTargets;

% Choose a training fucntion
switch trainingFunction
    case 'LM'
        trainFcn = 'trainlm';
    case 'BR'
        trainFcn = 'trainbr';
    case 'BFG'
        trainFcn = 'trainbfg';
    case 'RP'
        trainFcn = 'trainrp';
    case 'SCG'
        trainFcn = 'trainscg';
    case 'CGB'
        trainFcn = 'traincgb';
    case 'CGF'
        trainFcn = 'traincgf';
    case 'CGP'
        trainFcn = 'traincgp';
    case 'OSS'
        trainFcn = 'trainoss';
    case 'GDX'
        trainFcn = 'traingdx';
    otherwise
        disp('Choose the training methods proprely')
end

% Choose a layer size
switch netSize
    case 'small'
        hiddenLayerSize = [20, 10, 5];
    case 'mid'
        hiddenLayerSize = [40, 20, 10];
    case 'big'
        hiddenLayerSize = [60, 30, 15];
    otherwise
        disp('Choose the size of network proprely')
end

% Create a network object
numLayers = numel(hiddenLayerSize);
switch netType
    case 'feed-forward'
        net = patternnet(hiddenLayerSize, trainFcn, perfFunction);
        net.layers{numLayers+1}.transferFcn = 'tansig';
    case 'cascade'
        net = cascadeforwardnet(hiddenLayerSize,trainFcn);
        net.layers{numLayers+1}.transferFcn = 'tansig';
    case 'recurrent'
        layerDelays = 1:2;
        net = layrecnet(layerDelays,hiddenLayerSize,trainFcn);
        net.layers{numLayers+1}.transferFcn = 'tansig';
    otherwise
        disp('Choose the network type proprely');
end
net.performFcn = perfFunction;

% Transfer function of hidden layers
for i = 1:numLayers
    net.layers{i}.transferFcn = 'tansig';
end

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.1;
net.trainParam.max_fail = 10;
net.trainParam.epochs = 1000;

net_accuracy = zeros(numIterations, 1);
net_perfomance = zeros(numIterations, 1);
net_processing_time = zeros(numIterations, 1);

netOutput = {};
% while mean(net_accuracy) < 0.82  % To test many times
for numFeats = featsRange        % For 1:20 features testing
    cath_accuracy_arr = [];
    confMatrix = {};
    trainingTime = []; 
    
    % Get the truncated data set
    x = X(:,ranking(1:numFeats));
    x = RobustNormalization(x, 'std', 1);
    % t = rescale(Y, -1, 1);
    t = Y';
    x = x';
    
    for k = 1:numIterations 

        tic;
        net.inputs{1}.size = numFeats; 
        if isUseGPU == 1 
            [net,tr] = train(net, x, t, 'useGPU','yes');
        else
            [net,tr] = train(net, x, t);
        end
        pTime = toc;
        
        % Test the Network
        y = net(x);
        e = gsubtract(t, y);
        performance = perform(net, t, y);
        tind = vec2ind(t);
        yind = vec2ind(y);
        percentErrors = sum(tind ~= yind)/numel(tind);

        % Training Confusion Plot Variables
        yTrn = net(x(:,tr.trainInd));
        tTrn = t(:,tr.trainInd);
        % Validation Confusion Plot Variables
        yVal = net(x(:,tr.valInd));
        tVal = t(:,tr.valInd);
        % Test Confusion Plot Variables
        yTst = net(x(:,tr.testInd));
        tTst = t(:,tr.testInd);
        % Overall Confusion Plot Variables
        yAll = net(x);
        tAll = t;

        % Get the results
        [~, tempConfMatrix] = confusion(tAll,yAll);
        tempConfMatrix = tempConfMatrix';
        cathAccuracy = tempConfMatrix(2,2)/sum(tempConfMatrix(:,2));
        net_accuracy(k,1) = cathAccuracy;
        net_perfomance(k,1) = perform(net,t,y);
        net_processing_time(k,1) = pTime;
                
        trainingTime = cat(1, trainingTime, pTime); 
        confMatrix = cat(1, confMatrix, tempConfMatrix);
        cath_accuracy_arr = cat(1, cath_accuracy_arr, cathAccuracy);

        fprintf('%.d iteration (%s): accuracy = %.2f%%, perfomance = %.3f, time = %.2f sec\n', ...
                k, selection_method, 100*cathAccuracy, perform(net,t,y), pTime);
    end
    fprintf('Average accuracy: %.1f±%.1f%%\n', 100*mean(cath_accuracy_arr), 100*std(cath_accuracy_arr));
       
    % Writing all results into one variable
    if size(featsRange, 2) == 1
        netOutput{1, 1} = cath_accuracy_arr;
        netOutput{1, 2} = confMatrix;
        netOutput{1, 3} = net_perfomance;
        netOutput{1, 4} = trainingTime;    
    else
        netOutput{numFeats, 1} = cath_accuracy_arr;
        netOutput{numFeats, 2} = confMatrix;
        netOutput{numFeats, 3} = net_perfomance;
        netOutput{numFeats, 4} = trainingTime;
    end
end     % For 1:20 features testing

% end   % To test many times

if isSave == 1
    netOutputName = strcat(selection_method, '_', num2str(numFeats), '_features.mat'); 
    cd('Net methods comparison');
    save(netOutputName, 'netOutput');
    cd ..\
end

% Write to XLSX file

if size(featsRange, 2) == 1   
    % Perfomance
    xlsData = mean(netOutput{1,3});
    xlsData = cat(1, xlsData, std(netOutput{1,3}));
    % Zero string
    xlsData = cat(1, xlsData, 0);
    % Training time
    xlsData = cat(1, xlsData, mean(netOutput{1,4}));
    xlsData = cat(1, xlsData, std(netOutput{1,4}));
    % Raw accuracy
    xlsData = cat(1, xlsData, netOutput{1,1});
    xlswrite('netSingleTest.xlsx', xlsData, 1, 'B2');
    winopen('netSingleTest.xlsx')
else
    xlsData = [];
    for i = 1:size(netOutput,1)
        tempAccuracy = mean(netOutput{i, 1}); 
        xlsData = cat(1, xlsData, tempAccuracy);
    end
    xlswrite('netMultipleTest.xlsx', xlsData, 1, 'B2');
    winopen('netMultipleTest.xlsx')
end

% Plot Confusion
if isVisual == 1
%     figure, plotconfusion(tTrn, yTrn, 'Training', ...
%                           tVal, yVal, 'Validation', ...
%                           tTst, yTst, 'Test', ...
%                           tAll, yAll, 'Overall')
    figure, plotconfusion(t,y)
    % Uncomment these lines to enable various plots.
    % figure, plotperform(tr)
    % figure, plottrainstate(tr)
    % figure, ploterrhist(e)
    % figure, plotroc(t,y)
end


