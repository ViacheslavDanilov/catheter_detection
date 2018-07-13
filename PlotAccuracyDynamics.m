clear all; close all; clc;
set(0, 'DefaultFigureWindowStyle', 'normal');
currentFolder = pwd;
addpath(genpath(pwd));
%
span = 0.05;
smoothMethod = 'sgolay'; % 'lowess', 'loess', 'sgolay', 'rlowess', 'rloess', 'moving'

trainingType = 'net'; % svm, svm_uci, net
if strcmp(trainingType, 'svm') 
    sheet = 'FS comparison (SVM)';
    xlRange = 'C147:P166';
elseif strcmp(trainingType, 'net') 
    sheet = 'FS comparison (NET)';
    xlRange = 'C118:P137';
elseif strcmp(trainingType, 'svm_uci') 
    sheet = 'FS comparison (SVM UCI)';
    xlRange = 'C157:P186';
end

rawAccuracy = xlsread('Feature engineering (not separated).xlsm', sheet, xlRange);
rawFeatureStep = (1:1:size(rawAccuracy, 1))';
smoothFeatureStep = (1:0.1:size(rawAccuracy, 1))';
interpRawAccuracy = interp1(rawFeatureStep, rawAccuracy, smoothFeatureStep, 'pchip');

numRows = size(interpRawAccuracy, 1);
numCols = size(interpRawAccuracy, 2); 
smoothAccuracy = zeros(numRows, numCols);
for i = 1:numCols
    rawAccuracyVector = interpRawAccuracy(:,i);
    smoothAccuracyVector = smooth(rawAccuracyVector, span, smoothMethod);
    smoothAccuracy(:,i) = smoothAccuracyVector;
end

% DrawPlot(rawFeatureStep, rawAccuracy);   % raw data
DrawPlot(smoothFeatureStep, interpRawAccuracy);   % interpolated data 
% DrawPlot(smoothFeatureStep, smoothAccuracy);  % smoothed data

%%
plot(rawFeatureStep, rawAccuracy, '--', smoothFeatureStep, smoothAccuracy, '-');
%%
function DrawPlot(xData, yData)
    % Create figure
    figure1 = figure;
    scrSz = get(0, 'Screensize');
    set(gcf, 'Position', scrSz, 'Color', 'w');
    % Create axes
%     axes1 = axes('Parent',figure1);
%     axes1 = axes('ColorOrder',brewermap(14,'Set2'), 'NextPlot','replacechildren');
    axes1 = axes('ColorOrder', brewermap(14,'*Spectral'), 'Parent', figure1); 
    hold(axes1, 'on');

    plot1 = plot(xData, yData*100, 'linewidth', 1.5, 'MarkerSize', 4, 'Parent', axes1);
    
    set(plot1(1),'DisplayName','ILFS');
    set(plot1(2),'DisplayName','INFS');
    set(plot1(3),'DisplayName','ECFS');
    set(plot1(4),'DisplayName','MRMR');
    set(plot1(5),'DisplayName','RFFS');
    set(plot1(6),'DisplayName','MIFS');
    set(plot1(7),'DisplayName','FSCM');
    set(plot1(8),'DisplayName','LSFS');
    set(plot1(9),'DisplayName','MCFS');
    set(plot1(10),'DisplayName','UDFS');
    set(plot1(11),'DisplayName','CFS');
    set(plot1(12),'DisplayName','BDFS');
    set(plot1(13),'DisplayName','OFS');
    set(plot1(14),'DisplayName','ADFS');


    % Create ylabel
    textSize = 16;
    ylabel('Accuracy, %','FontSize',textSize);

    % Create xlabel
    xlabel('Number of features','FontSize',textSize);

    % Uncomment the following line to preserve the X-limits of the axes
    xlim(axes1,[0 xData(end, 1)+1]);
    % Uncomment the following line to preserve the Y-limits of the axes
    ylim([-1 100]);
    box(axes1,'on');
    grid(axes1,'on');
    % Set the remaining axes properties
    set(axes1,'FontName','Times New Roman','FontSize',textSize,'XMinorGrid','on',...
        'YMinorGrid','off','ZMinorGrid','on');
    % Create legend
    legend1 = legend(axes1,'show');
    set(legend1,'FontSize',textSize,'EdgeColor',[0 0 0],'Location','best');

    grid on
%     grid minor
    % grid toggles
%     title('PCHIP Interpolation');
end


