% plot basic bathymetry smaller estuary models
% Last edit: June 2024
% Author: Anne Baar

%NOG CHECKEN: T1 = 0JAAR


clear all; close all;clc;

%% paths
addpath('/home/618467/delft3dmatlab')
pathmodels = ('/home/618467/Dredging_final_runs/');
models= dir('/home/618467/Dredging_final_runs/L*');    % all folders in the folder (CHANGE TO YOUR LOCATION)
numfiles = length(models);


%% input
modelspl =[1,8,2,4,6]; %1:numfiles;%
timestep = 28; %15 = 50 years, 30 = 100 years
c1 = length(modelspl); % which models to compare (c1-c2)
c2 = 1;
pl = 1;

% model scenarios
season = [0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1];
locations = [0,12,12,12,12,12,12,12,12 , 24,24,24,24,24,24,24,24, 6,6,6,6,6,6,6,6];
intensity = [0,1000,1000,2000,2000,4000,4000,500,500, 1000,1000,2000,2000,4000,4000,500,500,1000,1000,2000,2000,4000,4000,8000,8000 ];
bin_intensity = [0,2,2,3,3,4,4,1,1,2,2,3,3,4,4,1,1,2,2,3,3,4,4,5,5];
tot_max_volume = intensity.*locations; % per morph year

DredgeX = [14.9583469498068	15.6249952295646	16.2916480747603	16.8750107654647	17.4583351559958	18.1249989746626	18.7083290057733	19.3749867417732	20.0416749054999	20.6249747267343	21.2083447088986	21.8750076840736	22.4583388092102	23.1249957738927	23.7916569723980	24.3750284437709	24.9583269370604	25.6250164747884	26.2916725980404	26.8749599300539	27.5416668608488	28.1250202328152     2.8708e+01    2.9375e+01];
DredgeY = [7.73790322580645	7.73790322580645	7.66532258064516	7.66532258064516	7.33467741935484	7.33467741935484	7.31854838709678	7.31854838709678	7.72177419354839	7.72177419354839	7.61693548387097	7.61693548387097	7.72177419354839	7.72177419354839	7.55241935483871	7.55241935483871	7.68951612903226	7.68951612903226	7.40725806451613	7.40725806451613	7.56854838709678	7.56854838709678     7.3105    7.5121];

% model characteristics
GRID = wlgrid('read','gridmuriel');
Lsea = 48;
slope = 0.05;%1e-4;
por = 1600./2650; % dry bed density/ density (= 1-porosity)
 DT = 3.4223; % years between saved timesteps

% color map for plots
cmap = [0.12156862745098039, 0.4666666666666667,  0.7058823529411765
    0.17254901960784313, 0.6274509803921569,  0.17254901960784313 % groen
    1.0,                 0.4980392156862745,  0.054901960784313725 % oranje
    %     0.8392156862745098,  0.15294117647058825, 0.1568627450980392  % rood
    0.5803921568627451,  0.403921568627451,   0.7411764705882353
    0.5490196078431373,  0.33725490196078434, 0.29411764705882354
];


%% calc
t0 = timestep;

DPS = zeros(length(timestep),length(GRID.X(1,:))+1,length(GRID.X)+1);


dredgeV = zeros(1,length(modelspl));
input = dredgeV;
EstuaryW = zeros(length(models),length(GRID.X)+1);
diff_cross = EstuaryW;

for k = 1:length(modelspl) %add 4th dimension if you also want to save timesteps
    
    path = ([pathmodels,models(modelspl(k)).name,'/']);%hydrorun of timestep of interest
    trih = vs_use([path 'trih-estuary.dat'],[path 'trih-estuary.def']);
    trim = vs_use([path 'trim-estuary.dat'],[path 'trim-estuary.def']);
    
    
    
    
    [XZ,YZ,MORFAC,TMAX,TMAX_tot] = loadtrim(trim);
    morf_time = round(vs_let(trim,'map-infavg-serie',{t0:t0},'MFTAVG','quiet')./365.25)-round(vs_let(trim,'map-infavg-serie',{1:1},'MFTAVG','quiet')./365.25); %output model = days since start simulation!
    DPS(k,:,:) = squeeze(vs_let(trim,'map-sed-series',{t0:t0},'DPS',{1:size(XZ,1) 1:size(XZ,2)},'quiet')*-1);
    active = squeeze(vs_let(trim,'map-series',{t0:t0},'KFU',{1:size(XZ,1) 1:size(XZ,2)},'quiet'))+squeeze(vs_let(trim,'map-series',{t0:t0},'KFV',{1:size(XZ,1) 1:size(XZ,2)},'quiet'));
    
    DPS(k,1,:) = NaN;
    DPS(k,end,:) = NaN;
    DPS(k,:,1) = NaN;
    DPS(k,:,end) = NaN;
    
    test = squeeze(DPS(k,:,:));
    
    DPSmask = ones(size(test));
    DPSmask(1,:) = 0;
    DPSmask(end,:) = 0;
    DPSmask(:,1) = 0;
    DPSmask(:,end) = 0;
    DPSmask(isnan(active)) = 0;
    %     DPSmask(DPS>(S-0.08)) = 0; %originally 0.08
    BW1 = bwareaopen((DPSmask),10,4);
    BW2 = imfill(BW1,26,'holes');
    DPSmask = bwareaopen((BW2),500,4);
    mask = DPSmask;
    DPSmask = test;
    DPSmask(mask == 0) = NaN;
    
    XZ2 = XZ./1000;
    YZ2 = YZ./1000;
    xas = XZ2(100,:);
    
    
    areatot = 0;
    areacel = zeros(size(squeeze(DPS(k,:,:))));
    test2 = ones(size(XZ,1),size(XZ,2));
    for l = 1:size(XZ,1)-1
        for m = 1:size(XZ,2)-1
            v0 = [XZ(l,m)     YZ(l,m)     test2(l,m)    ];
            v1 = [XZ(l,m+1)   YZ(l,m+1)   test2(l,m+1)  ];
            v2 = [XZ(l+1,m)   YZ(l+1,m)   test2(l+1,m)  ];
            v3 = [XZ(l+1,m+1) YZ(l+1,m+1) test2(l+1,m+1)];
            a = (v1 - v0);
            b = (v2 - v0);
            c = (v3 - v0);
            A = 1/2*(norm(cross(a,c)) + norm(cross(b,c)));
            areacel(l,m) = A;
        end
        
    end
    
    areacel(1,:) = NaN;
    areacel(end,:) = NaN;
    areacel(:,1) = NaN;
    areacel(:,end) = NaN;
    areacel(2,:) = NaN;
    areacel(end-1,:) = NaN;
    areacel(:,2) = NaN;
    areacel(:,end-1) = NaN;
    ind = active>0;
    Yind = YZ.*ind;
    Yind(Yind ==0)=NaN;
    EstuaryW(k,:) = nanmax(Yind)-nanmin(Yind); % in km
    

        
    
    
    
    %     scatter(DredgeX(dredge_locations),3*ones(1,length(DredgeX(dredge_locations))),30,'k','*')
    %     xlim([10,30])
    cummQb = vs_let(trih,'his-sed-series',{t0},'SBTRC');
    input(k) = round(-cummQb(16).*400);
    
    if modelspl(k)>1
        maxvol = 2049.1 *morf_time; %per location with pores
        dredgeVo = vs_let(trih,'his-dad-series',{15:15},'DREDGE_VOLUME');
        dredgeVot = vs_let(trih,'his-dad-series',{1:t0},'DREDGE_VOLUME');
        extravol = maxvol-dredgeVo;
        dredgeV(k) = nansum(round(dredgeVo.*por)); %sediment withour pore space
        
            
        diffbedlevel = squeeze(DPS(k,:,:))-squeeze(DPS(1,:,:));
        diffvolume = diffbedlevel.*areacel;
        diff_cross(k,:) = nansum(diffvolume);
        
        %     InstQb = vs_let(trih,'his-sed-series',{1:t0},'SBTR');
        
        %test
        
        dredgeVoALL = vs_let(trih,'his-dad-series',{1:t0},'DREDGE_VOLUME');
        dredgeVoALL_year = diff(dredgeVoALL)./(4500*400/60/24/365.25);
        dredgelimit = ([1:length(dredgeVoALL(:,1))]-2)'.*(4500*400/60/24/365.25).*2049.1.*ones(length(dredgeVoALL(:,1)),1); %incl. spinup interval
        Vnotdredged = dredgelimit-dredgeVoALL ;
        
        % figure(100*k);plot(dredgeVoALL_year(2:end,:))
%         figure(44); hold on; plot(nansum(dredgeVot,2)./(4500*400/60/24/365.25),'color',cmap(bin_intensity(modelspl((k))),:),'Linewidth',1.5)
    end
    
    
    % calc
    
    
    
    
end



fraction = round(dredgeV./input*100)./100;



%% figures
% close all
if length(modelspl)<6
    sm = 3;
    for i3 = 2:length(modelspl)
%         diffbedlevel = squeeze(DPS(i3,:,:))-squeeze(DPS(1,:,:));
%         diffvolume = diffbedlevel.*areacel;
%         diff_cross = nansum(diffvolume);
%         % diff_cross(diff_cross>10) = NaN;
        
        figure(78); hold on; plot(xas(2:end-1),smooth(diff_cross(i3,2:end-1) ,sm),'color',cmap(i3-1,:),'LineWidth',1.5)
    end
    xlim([10,30])
          ylim([-6,3].*10^4)
    ylabel('relative total bedlevel change over the cross-section [m]')
    
    %
    
    for i3 = [1:length(modelspl),1]
        
        if i3 ==1
            figure(77); hold on; plot(xas(2:end-1),smooth(EstuaryW(i3,2:end-1) ,sm),'color','k','LineWidth',1.2)
            
        else
            figure(77); hold on; plot(xas(2:end-1),smooth(EstuaryW(i3,2:end-1) ,sm),'color',cmap(i3-1,:),'LineWidth',1.5)%bin_intensity(modelspl((i3)))
        end
    end
    xlim([10,30])
          ylim([0,2000])
    ylabel('estuary width [m]')
    legend({'control model','12 locations, 500 m3/y','12 locations, 1000 m3/y','12 locations , 2000 m3/y','12 locations, 4000 m3/y'},'box','off','fontsize',12)
    
    %
    depthC = prctile(squeeze(DPS(1,:,:)),5);
    
    for i3 = [1:length(modelspl),1]
        depth = prctile(squeeze(DPS(i3,:,:)),5);
        
        if i3 == 1
            figure(76); hold on; plot(xas(2:end-1),smooth(depth(2:end-1) ,sm),'color','k','LineWidth',1.2)
        else
            figure(76); hold on; plot(xas(2:end-1),smooth(depth(2:end-1) ,sm),'color',cmap(i3-1,:),'LineWidth',1.5)
        end
    end
    xlim([10,30])
    %       ylim([-10,10])
    ylabel('95th depth percentile [m]')
    %
    %
    
    %
    %
    
end



%% data reductie proberen....
% fraction = dredge volume / river supply
% season = [0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1];
% locations = [0,12,12,12,12,12,12,12,12 , 24,24,24,24,24,24,24,24, 6,6,6,6,6,6,6,6];
% intensity = [0,1000,1000,2000,2000,4000,4000,500,500, 1000,1000,2000,2000,4000,4000,500,500,1000,1000,2000,2000,4000,4000,8000,8000 ];
% bin_intensity =  [0,2,2,3,3,4,4,1,1,2,2,3,3,4,4,1,1,2,2,3,3,4,4,5,5];
% tot_max_volume = intensity.*locations; % per morph year

totWch = sum(EstuaryW(:,2:end-1) - EstuaryW(1,2:end-1),2);
diffT = sum(diff_cross,2);
 
lpl = locations./6;
lpl(lpl ==4) = 3;

figure;
hold on
for i = 1:5
scatter(2000,2000,50,cmap(i,:),'filled')
end
scatter(2000,2000,50,'k','filled')
scatter(2000,2000,50,'k','filled','marker','^')


for i = 2:length(totWch)
    if season(i) ==0
scatter(lpl(i),totWch(i),50,cmap(bin_intensity(i),:),'filled')
    else
        scatter(lpl(i),totWch(i),50,cmap(bin_intensity(i),:),'filled','marker','^')
    end
    
end

xlim([0.5,3.5])
set(gca,'Xtick',[1,2,3],'xticklabel',[6,12,24])
xlabel('dredging locations')
ylabel ('total width change')
nn=legend({'500 m3/y','1000 m3/y','2000 m3/y','4000 m3/y','8000 m3/y','contineous','seasonal'},'box','off','fontsize',12,'location','Northwest');

%
figure;
hold on
for i = 1:5
scatter(2000,2000,50,cmap(i,:),'filled')
end
scatter(2000,2000,50,'k','filled')
scatter(2000,2000,50,'k','filled','marker','^')

diffT(diffT>0)=NaN;
for i = 2:length(totWch)
    if season(i) ==0
scatter(lpl(i),diffT(i),50,cmap(bin_intensity(i),:),'filled')
    else
        scatter(lpl(i),diffT(i),50,cmap(bin_intensity(i),:),'filled','marker','^')
    end
    
end

xlim([0.5,3.5])
set(gca,'Xtick',[1,2,3],'xticklabel',[6,12,24])
xlabel('dredging locations')
ylabel ('total volume change')
nn=legend({'500 m3/y','1000 m3/y','2000 m3/y','4000 m3/y','8000 m3/y','contineous','seasonal'},'box','off','fontsize',12,'location','eastoutside');

figure
hold on
for i = 1:5
scatter(2000,2000,50,cmap(i,:),'filled')
end
scatter(2000,2000,50,'k','filled')
scatter(2000,2000,50,'k','filled','marker','^')


for i = 2:length(totWch)
    if season(i) ==0
scatter(fraction(i),diffT(i),50,cmap(bin_intensity(i),:),'filled')
    else
        scatter(fraction(i),diffT(i),50,cmap(bin_intensity(i),:),'filled','marker','^')
    end
    
end

xlim([0,2])
% set(gca,'Xtick',[1,2,3],'xticklabel',[6,12,24])
xlabel('dredged volume / river supply')
ylabel ('total volume change')
nn=legend({'500 m3/y','1000 m3/y','2000 m3/y','4000 m3/y','8000 m3/y','contineous','seasonal'},'box','off','fontsize',12,'location','eastoutside');


%
figure
hold on
for i = 1:5
scatter(2000,2000,50,cmap(i,:),'filled')
end
scatter(2000,2000,50,'k','filled')
scatter(2000,2000,50,'k','filled','marker','^')


for i = 2:length(totWch)
    if season(i) ==0
scatter(fraction(i),totWch(i),50,cmap(bin_intensity(i),:),'filled')
    else
        scatter(fraction(i),totWch(i),50,cmap(bin_intensity(i),:),'filled','marker','^')
    end
    
end

xlim([0,2])
% set(gca,'Xtick',[1,2,3],'xticklabel',[6,12,24])
xlabel('dredged volume / river supply')
ylabel ('total width change')
nn=legend({'500 m3/y','1000 m3/y','2000 m3/y','4000 m3/y','8000 m3/y','contineous','seasonal'},'box','off','fontsize',12,'location','eastoutside');
