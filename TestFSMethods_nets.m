% clear all; close all; clc;
set(0, 'DefaultFigureWindowStyle', 'normal');
currentFolder = pwd;
addpath(genpath(pwd));
warning('off', 'all');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initial Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numFeats = 3;
numIterations = 20;
isUseGPU = 1;
netType = 'cascade';   % 'feed-forward', 'cascade', 'recurrent'
netSize = 'mid';

if isUseGPU == 1
    trainingFunction = 'SCG';
    perfFunction = 'crossentropy';
else
    trainingFunction = 'BR';
    perfFunction = 'mse';
end

isVisual = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

listFS = {'ILFS', 'LSFS', 'UDFS', 'MIFS', 'FSCM', 'MCFS', 'OFS', 'PDF ADFS'};
[methodID] = readInput(listFS);
selectedMethod = listFS{methodID};

% Get ranking
switch selectedMethod
    case 'ILFS'
        xlRange = 'C67:C86';
    case 'LSFS'
        xlRange = 'J67:J86';
    case 'UDFS'
        xlRange = 'L67:L86';    
    case 'MIFS'
        xlRange = 'H67:H86';
    case 'FSCM'
        xlRange = 'I67:I86';
    case 'MCFS'
        xlRange = 'K67:K86';
    case 'OFS'
        xlRange = 'O67:O86';    
    case 'PDF ADFS'
        xlRange = 'P67:P86';
    otherwise
        disp('Unknown method.')
end

dataFile = 'Feature engineering (not separated).xlsm';
sheet = 'FS comparison (SVM)';
ranking = xlsread(dataFile, sheet, xlRange);

% Load the data
x = load('inputs (not separated).mat');
y = load('targets (not separated).mat');
X = x.netTrainInputs;
Y = y.netTrainTargets;

% Get the truncated data set
x = X(:,ranking(1:numFeats));
x = RobustNormalization(x, 'std', 1);
% t = rescale(Y, -1, 1);
t = Y';
x = x';

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
% while mean(net_accuracy) < 0.4838  % To test many times
    rng(round(rand*10))
    for k = 1:numIterations
        tic;
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
        [~, confMatrix] = confusion(tAll,yAll);
        confMatrix = confMatrix';
        cathAccuracy = confMatrix(2,2)/sum(confMatrix(:,2));
        net_accuracy(k,1) = cathAccuracy;
        net_perfomance(k,1) = perform(net,t,y);
        net_processing_time(k,1) = pTime;
        fprintf('%.d iteration (%s): accuracy = %.2f%%, perfomance = %.3f, time = %.2f sec\n', ...
                k, selectedMethod, 100*cathAccuracy, perform(net,t,y), pTime);
    end
    fprintf('Average accuracy: %.1f±%.1f%%\n', 100*mean(net_accuracy), 100*std(net_accuracy));
    
% end   % To test many times

fprintf('Final accuracy: %.1f±%.1f%%\n', 100*mean(net_accuracy), 100*std(net_accuracy));
% Save data
methodName = strcat(selectedMethod, '_', ...
                    num2str(numFeats), '_features_', ...
                    netType, '_', ...
                    netSize, '.mat'); 
cd('Net methods comparison');
save(methodName, 'net_accuracy', 'net_perfomance', 'net_processing_time');
cd ..\

% Write to XLSX file
xls_filename = 'testdata3.xlsx';
tempInfo = mean(net_perfomance);
tempInfo = cat(1, tempInfo, std(net_perfomance));
tempInfo = cat(1, tempInfo, 0);
tempInfo = cat(1, tempInfo, mean(net_processing_time));
tempInfo = cat(1, tempInfo, std(net_processing_time));
tempInfo = cat(1, tempInfo, net_accuracy);
sheet = 1;
cols_xlRange = 'B2';
xlswrite(xls_filename, tempInfo, sheet, cols_xlRange);

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

winopen('testdata3.xlsx')
