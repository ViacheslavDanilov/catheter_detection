
x = 1:1:20;
xlRange = 'C4:P23';% 'C4:P23', 'C27:P46', 'C50:P69';
y = xlsread('Feature engineering (not separated).xlsm', 'FS comparison (SVM) (2)', xlRange);
xq = 1:0.01:20;
yq6 = interp1(x,y,xq,'pchip');
DrawPlot(xq, yq6);
%%
% x = 1:1:20;
% x = x';
% y = xlsread('Feature engineering (not separated).xlsm', 'FS comparison (SVM) (2)', 'E4:G23');
% xq = 1:0.01:20;
% yq6 = interp1(x,y,xq,'pchip');
% plot(x,y,'--', xq, yq6, '-');
%%
function DrawPlot(Xdata, YMatrix)
    % Create figure
    figure1 = figure;

    % Create axes
    axes1 = axes('Parent',figure1);
    hold(axes1,'on');

    % Create multiple lines using matrix input to plot
    plot1 = plot(Xdata,YMatrix,'MarkerSize',3,'Parent',axes1);
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
    set(plot1(14),'DisplayName','PDF ADFS');

    % Create ylabel
    ylabel('Accuracy, %','FontSize',14);

    % Create xlabel
    xlabel('Number of features','FontSize',14);

    % Uncomment the following line to preserve the X-limits of the axes
    xlim(axes1,[0 21]);
    % Uncomment the following line to preserve the Y-limits of the axes
    ylim(axes1,[0 1]);
    box(axes1,'on');
    grid(axes1,'on');
    % Set the remaining axes properties
    set(axes1,'FontName','Times New Roman','FontSize',14,'XMinorGrid','on',...
        'YMinorGrid','off','ZMinorGrid','on');
    % Create legend
    legend1 = legend(axes1,'show');
    set(legend1,'FontSize',12,'EdgeColor',[0 0 0],'Location','best');

%     figure
    yq6 = interp1(x,y,xq,'pchip');
    % plot(x, y, 'o', xq, yq6, ':.', 'MarkerSize', 3); % 2 lines
    plot(xq, yq6, '-', 'MarkerSize', 3);
    legend('ILFS', 'INFS', 'ECFS', 'MRMR', 'RFFS', ...
    'MIFS', 'FSCM', 'LSFS', 'MCFS', ...
    'UDFS', 'CFS', 'BDFS', 'OFS','PDF ADFS');
    ylim([0 1]);
    xlim([0 21]);
    grid on
    grid minor
    % grid toggles
    title('PCHIP Interpolation');
end


