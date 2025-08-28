% smaller estuary models bathymetry analyses
% dredging runs
% Last edit: October 2024
% Author: Anne Baar

clear all
close all
clc

DT = 3.4223; % years between saved timesteps

% plot properties
runs = [26,29,30] ;%; %runs for plot   24: [19,21,12,14,15], [21,14,15]    12: [2,4,5,7,8],  [4,7,8]  6:[24,26,27,29,30]
nrloc = 6; % nr of locations (plot title)


cmap = [0.12156862745098039, 0.4666666666666667,  0.7058823529411765
    0.17254901960784313, 0.6274509803921569,  0.17254901960784313 % groen
    1.0,                 0.4980392156862745,  0.054901960784313725 % oranje
    0.8392156862745098,  0.15294117647058825, 0.1568627450980392  %rood
    0.5803921568627451,  0.403921568627451,   0.7411764705882353  %paars
    0.5490196078431373,  0.33725490196078434, 0.29411764705882354 ];
fs = 12; % font size

mk = {'x','v','o','^','d'};
%% load model output files
addpath('C:\Users\awbaar\Documents\Delft3D models\smaller estuary\Dredging final runs\matlab output')
load("XZ.mat");load("YZ.mat"); load("morf_time.mat"); load("areacells.mat")

morphology_files = dir('C:\Users\awbaar\Documents\Delft3D models\smaller estuary\Dredging final runs\matlab output\L*');
numruns = length(morphology_files);
morphology_output = cell(1,numruns );

for i1 = 1:numruns
    morphology_output{i1} = load(morphology_files(i1).name);
end

%% bathymetry changes
areacel = reshape(areacel,1,162,290);
morph_change = diff(morphology_output{1}.bathymetry).*areacel;       % control scenario: difference in bedlevels between timesteps * area cell = volume change

totalbedlevelchangeSea = zeros(numruns,28);
totalbedlevelchangeDelta = zeros(numruns,28);
totalbedlevelchangeDA= zeros(numruns,28);
fraction= zeros(numruns,1); % sediment dredged / sediment supply by river


% zones
totalbedlevelchangeSea(1,:) = nansum(nansum(morph_change(2:end,:,1:48),2),3)'./DT; % control run
totalbedlevelchangeDelta(1,:) = nansum(nansum(morph_change(2:end,:,49:108),2),3)'./DT;
totalbedlevelchangeDA(1,:) = nansum(nansum(morph_change(2:end,:,108:end),2),3)'./DT;

for i2 = 2:numruns %dredging scenarios
    morph_run = morphology_output{i2}.bathymetry;
    active_run = morphology_output{i2}.activecells;
    
    % filter for model disturbances 
    morph_run(morph_run>5.01) = NaN;
    morph_changeD = diff(morph_run).*areacel;      % sandmining scenario

    totalbedlevelchangeSea(i2,:) = nansum(nansum(morph_changeD(2:end,:,1:48),2),3)'./3.4223; % ~ per year (dt = 3.4223 years)
    totalbedlevelchangeDelta(i2,:) = nansum(nansum(morph_changeD(2:end,:,49:108),2),3)'./3.4223;
    totalbedlevelchangeDA(i2,:) = nansum(nansum(morph_changeD(2:end,:,108:end),2),3)'./3.4223;

    fraction(i2,1) = max(morphology_output{i2}.fraction_totaldredged_sedimentinput);

end

prct_supply = num2str(round(fraction(runs).*100)); % for figure legend

%% figures
% relative to control run
figure % relative to control run
title([num2str(nrloc),' sand mining locations'])



hold on
% legend
plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{1})
plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{2})
plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{3})
plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{4})
plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{5})
plot([1000,1000],[0,0],'LineStyle','-','LineWidth',1.5,'Color',cmap(1,:))
plot([1000,1000],[0,0],'LineStyle','-','LineWidth',1.5,'Color',cmap(2,:))
plot([1000,1000],[0,0],'LineStyle','-','LineWidth',1.5,'Color',cmap(4,:))



n = 1;
for i3 = runs
if i3 == 6
    temp = totalbedlevelchangeDA(i3,:)-totalbedlevelchangeDA(1,:);
    temp(6) = NaN;
    plot(morf_time(3:end),totalbedlevelchangeSea(i3,:)-totalbedlevelchangeSea(1,:),'Color',cmap(1,:),'LineWidth',1.5,'Marker',mk{n})
    plot(morf_time(3:end),totalbedlevelchangeDelta(i3,:)-totalbedlevelchangeDelta(1,:),'Color',cmap(2,:),'LineWidth',1.5,'Marker',mk{n})
    plot(morf_time(3:end),temp,'Color',cmap(4,:),'LineWidth',1.5,'Marker',mk{n})
else
        plot(morf_time(3:end),totalbedlevelchangeSea(i3,:)-totalbedlevelchangeSea(1,:),'Color',cmap(1,:),'LineWidth',1.5,'Marker',mk{n})
    plot(morf_time(3:end),totalbedlevelchangeDelta(i3,:)-totalbedlevelchangeDelta(1,:),'Color',cmap(2,:),'LineWidth',1.5,'Marker',mk{n})
    plot(morf_time(3:end),totalbedlevelchangeDA(i3,:)-totalbedlevelchangeDA(1,:),'Color',cmap(4,:),'LineWidth',1.5,'Marker',mk{n})
end

n = n+1;
end

plot([0,100],[0,0],'LineStyle','-','Color','k')
plot([50,50],[-9e4,1.6e4],'LineStyle','--','Color','k')
box on
ylim([-9e4,1.6e4])
xlim([0,100])
xlabel('time [years]')
ylabel('Relative volume change /year [m^3]')
set(gca,'FontSize',fs)
legend({[prct_supply(1,:),'% of sediment supply'],[prct_supply(2,:),'% of sediment supply'],[prct_supply(3,:),'% of sediment supply'],[prct_supply(4,:),'% of sediment supply'],[prct_supply(5,:),'% of sediment supply'],'Sea basin','Area without mining','Area with mining'},'box','off','Location','southeast','FontSize',fs-1)

% legend({'Sea basin','Area without mining','Area with mining'},'box','off','Location','southeast','FontSize',fs)
text(10,-2.7e4,'Sand mining','FontSize',fs)
text(60,-2.7e4,'Recovery','FontSize',fs)
% 
% set(gcf,'PaperPositionMode','auto');
%  print(['C:\Users\awbaar\OneDrive - Delft University of Technology\Documents\papers\Dredging\figures\volumechange_',num2str(nrloc),'_sandmininglocations'],'-dpng','-r150');




%% growth rate
% 
% % close all
% figure % relative to control run
% title([num2str(nrloc),' sand mining locations'])
% 
% 
% 
% hold on
% % legend
% plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{1})
% plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{2})
% plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{3})
% plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{4})
% plot([1000,1000],[0,0],'LineStyle','-','LineWidth',1.5,'Color',cmap(1,:))
% plot([1000,1000],[0,0],'LineStyle','-','LineWidth',1.5,'Color',cmap(2,:))
% plot([1000,1000],[0,0],'LineStyle','-','LineWidth',1.5,'Color',cmap(5,:))
% 
% 
% 
% n = 1;
% for i3 = runs
%     totalbedlevelchangeDelta(1,abs(totalbedlevelchangeDelta(1,:))<500)=NaN;
% if i3 == 6
%     temp = (totalbedlevelchangeDA(i3,:)-totalbedlevelchangeDA(1,:))./abs(totalbedlevelchangeDA(1,:));
% %     temp(6) = NaN;
%     plot(morf_time(3:end),(totalbedlevelchangeSea(i3,:)-totalbedlevelchangeSea(1,:))./abs(totalbedlevelchangeSea(1,:)),'Color',cmap(1,:),'LineWidth',1.5,'Marker',mk{n})
%     plot(morf_time(3:end),(totalbedlevelchangeDelta(i3,:)-totalbedlevelchangeDelta(1,:))./abs(totalbedlevelchangeDelta(1,:)),'Color',cmap(2,:),'LineWidth',1.5,'Marker',mk{n})
%     plot(morf_time(3:end),temp,'Color',cmap(5,:),'LineWidth',1.5,'Marker',mk{n})
% else
%     plot(morf_time(3:end),(totalbedlevelchangeSea(i3,:)-totalbedlevelchangeSea(1,:))./abs(totalbedlevelchangeSea(1,:)),'Color',cmap(1,:),'LineWidth',1.5,'Marker',mk{n})
%     plot(morf_time(3:end),(totalbedlevelchangeDelta(i3,:)-totalbedlevelchangeDelta(1,:))./abs(totalbedlevelchangeDelta(1,:)),'Color',cmap(2,:),'LineWidth',1.5,'Marker',mk{n})
%     plot(morf_time(3:end),(totalbedlevelchangeDA(i3,:)-totalbedlevelchangeDA(1,:))./abs(totalbedlevelchangeDA(1,:)),'Color',cmap(5,:),'LineWidth',1.5,'Marker',mk{n})
% end
% 
% n = n+1;
% end
% 
% plot([0,100],[0,0],'LineStyle','-','Color','k')
% plot([50,50],[-7,2],'LineStyle','--','Color','k')
% box on
% xlim([0,100])
% ylim([-7,1])
% xlabel('Time [years]')
% ylabel('Rate of change [-]')
% set(gca,'FontSize',fs)
% legend({[prct_supply(1,:),'% of sediment supply'],[prct_supply(2,:),'% of sediment supply'],[prct_supply(3,:),'% of sediment supply'],[prct_supply(4,:),'% of sediment supply'],'Sea basin','Area without mining','Area with mining'},'box','off','Location','southeast','FontSize',fs-1)
% text(10,0.7,'Sand mining','FontSize',fs)
% text(60,0.7,'Recovery','FontSize',fs)
% 
% % set(gcf,'PaperPositionMode','auto');
% %  print(['C:\Users\awbaar\OneDrive - Delft University of Technology\Documents\papers\Dredging\figures\Changerate_',num2str(nrloc),'_sandmininglocations'],'-dpng','-r150');



%% old/extra

% figure
% title('12 locations, 2000m3/year')
% hold on
% plot(morf_time(3:end),totalbedlevelchangeSea./abs(totalbedlevelchangeSeaC),'Color',cmap(1,:),'LineWidth',2)
% plot(morf_time(3:end),totalbedlevelchangeDelta./abs(totalbedlevelchangeDelta),'Color',cmap(2,:),'LineWidth',2)
% plot(morf_time(3:end),totalbedlevelchangeDA./abs(totalbedlevelchangeDA),'Color',cmap(5,:),'LineWidth',2)
% plot([0,100],[0,0],'LineStyle','-','Color','k')
% plot([0,100],[-1,-1],'LineStyle',':','Color','k')
% plot([0,100],[1,1],'LineStyle',':','Color','k')
% plot([50,50],[-6,2],'LineStyle','--','Color','k')
% box on
% % ylim([-6e4,1.1e4])
% xlabel('time [years]')
% ylabel('volume change sandmining run / control run [-]')
% set(gca,'FontSize',fs)
% legend({'35% of sediment supply','76% of sediment supply','170 % of sediment supply','Sea basin','Area without mining','Area with mining'},'box','off','Location','southeast','FontSize',fs)
% text(10,1.5,'Sand mining','FontSize',fs)
% text(60,1.5,'Recovery','FontSize',fs)

%% relative to control run (ratio)
% figure % relative to control run
% title('12 sand mining locations')
% 
% 
% 
% hold on
% % legend
% plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{1})
% plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{2})
% plot([1000,1000],[0,0],'LineStyle','-','Color','k','Marker',mk{3})
% plot([1000,1000],[0,0],'LineStyle','-','LineWidth',1.5,'Color',cmap(1,:))
% plot([1000,1000],[0,0],'LineStyle','-','LineWidth',1.5,'Color',cmap(2,:))
% plot([1000,1000],[0,0],'LineStyle','-','LineWidth',1.5,'Color',cmap(5,:))
% 
% 
% 
% n = 1;
% for i3 = runs
% if i3 == 6
%     temp = totalbedlevelchangeDA(i3,:)./abs(totalbedlevelchangeDA(1,:));
%     temp(6) = NaN;
%     plot(morf_time(3:end),totalbedlevelchangeSea(i3,:)./abs(totalbedlevelchangeSea(1,:)),'Color',cmap(1,:),'LineWidth',1.5,'Marker',mk{n})
%     plot(morf_time(3:end),totalbedlevelchangeDelta(i3,:)./abs(totalbedlevelchangeDelta(1,:)),'Color',cmap(2,:),'LineWidth',1.5,'Marker',mk{n})
%     plot(morf_time(3:end),temp,'Color',cmap(5,:),'LineWidth',1.5,'Marker',mk{n})
% else
%     plot(morf_time(3:end),totalbedlevelchangeSea(i3,:)./abs(totalbedlevelchangeSea(1,:)),'Color',cmap(1,:),'LineWidth',1.5,'Marker',mk{n})
%     plot(morf_time(3:end),totalbedlevelchangeDelta(i3,:)./abs(totalbedlevelchangeDelta(1,:)),'Color',cmap(2,:),'LineWidth',1.5,'Marker',mk{n})
%     plot(morf_time(3:end),totalbedlevelchangeDA(i3,:)./abs(totalbedlevelchangeDA(1,:)),'Color',cmap(5,:),'LineWidth',1.5,'Marker',mk{n})
% end
% 
% n = n+1;
% end
% 
% plot([0,100],[0,0],'LineStyle','-','Color','k')
% plot([0,100],[-1,-1],'LineStyle',':','Color','k')
% plot([0,100],[1,1],'LineStyle',':','Color','k')
% plot([50,50],[-6,2],'LineStyle','--','Color','k')
% box on
% xlim([0,100])
% ylim([-6,2])
% xlabel('time [years]')
% ylabel('volume change sandmining run / control run [-]')
% set(gca,'FontSize',fs)
% 
% legend({[prct_supply(1,:),'% of sediment supply'],[prct_supply(2,:),'% of sediment supply'],[prct_supply(3,:),'% of sediment supply'],[prct_supply(4,:),'% of sediment supply'],'Sea basin','Area without mining','Area with mining'},'box','off','Location','southeast','FontSize',fs-1)
% text(10,1.5,'Sand mining','FontSize',fs)
% text(60,1.5,'Recovery','FontSize',fs)

%% absolute volume change between time steps
% figure
% title('12 locations, 2000m3/year')
% hold on
% for i3 = runs
% plot(morf_time(3:end),totalbedlevelchangeSea(i3,:),'Color',cmap(1,:),'LineWidth',2)
% plot(morf_time(3:end),totalbedlevelchangeDelta(i3,:),'Color',cmap(2,:),'LineWidth',2)
% plot(morf_time(3:end),totalbedlevelchangeDA(i3,:),'Color',cmap(5,:),'LineWidth',2)
% end
% plot(morf_time(3:end),totalbedlevelchangeSea(1,:),'LineStyle','--','Color',cmap(1,:),'LineWidth',2)
% plot(morf_time(3:end),totalbedlevelchangeDelta(1,:),'LineStyle','--','Color',cmap(2,:),'LineWidth',2)
% plot(morf_time(3:end),totalbedlevelchangeDA(1,:),'LineStyle','--','Color',cmap(5,:),'LineWidth',2)
% plot([0,100],[0,0],'LineStyle','-','Color','k')
% plot([50,50],[-1e5,2e5],'LineStyle','--','Color','k')
% 
% box on
% xlabel('time [years]')
% ylabel('Volume change /year [m^3]')
% set(gca,'FontSize',fs)
% text(10,150000,'Sand mining','FontSize',fs)
% text(60,150000,'Recovery','FontSize',fs)
% legend({'Sea basin','Non dredged area','Dredged area'},'box','off','Location','best','FontSize',fs)
