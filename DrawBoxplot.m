figure1 = figure;
axes1 = axes('Parent',figure1);
x1 = cath; x2 = tissue;
x = [x1;x2];
g = [ones(size(x1)); 2*ones(size(x2))];
boxplot(x,g,'Notch','off', 'MedianStyle', 'line', 'FullFactors', 'on', 'Widths',0.6)
ylim(axes1,[0.7 1.05]);
% xlabel('Correlation')
ylabel('Correlation');
box(axes1,'on');
set(axes1,'FontName','Times New Roman','FontSize', 36, 'TickLabelInterpreter',...
    'none','XTick',[1 2],'XTickLabel',{'Catheter','Tissue'},'YGrid', 'on');