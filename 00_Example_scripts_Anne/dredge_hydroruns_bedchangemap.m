%% Hydrodynamic characteristics along the estuary
% last edit Nov 2024
% Anne Baar
% plot variables along estuary length
%%
clear all; close all;clc;

addpath('/home/618467/delft3dmatlab')
pathmodels = ('/home/618467/Dredging_final_runs/hydroruns/');
models= dir('/home/618467/Dredging_final_runs/hydroruns/L*');    % all folders in the folder (CHANGE TO YOUR LOCATION)
numfiles = length(models);
modelpl = [2,23,26,29];%1:numfiles;%[3,6,9,12];%[3,24,6,21]-1;%3:3:27;
ycel = 162;
xcel = 290;


sm = 150;

cmap = [0.12156862745098039, 0.4666666666666667,  0.7058823529411765
    0.17254901960784313, 0.6274509803921569,  0.17254901960784313 % groen
    1.0,                 0.4980392156862745,  0.054901960784313725 % oranje
    0.8392156862745098,  0.15294117647058825, 0.1568627450980392
    0.5803921568627451,  0.403921568627451,   0.7411764705882353
    0.5490196078431373,  0.33725490196078434, 0.29411764705882354
    0.8901960784313725,  0.4666666666666667,  0.7607843137254902
    0.4980392156862745,  0.4980392156862745,  0.4980392156862745
    0.7372549019607844,  0.7411764705882353,  0.13333333333333333
    0.09019607843137255, 0.7450980392156863,  0.8117647058823529];
%% calculation
n = 1;
n2 = 1;



for k = modelpl
    
    path = ([pathmodels,models(k).name,'/']);
    
    trim = vs_use([path 'trim-estuary.dat'],[path 'trim-estuary.def']);
    
    
    if n == 1
        XZ = squeeze(vs_let(trim,'map-const','XZ','quiet'))./1000;
        XZp = XZ(100,:);
        YZ = squeeze(vs_let(trim,'map-const','YZ','quiet'))./1000;
    end
    %% flow velocity
    bedlevelb = squeeze(vs_let(trim,'map-sed-series',{1:1},'DPS',{1:ycel 1:xcel},'quiet')*-1);
    bedlevele = squeeze(vs_let(trim,'map-sed-series',{124:124},'DPS',{1:ycel 1:xcel},'quiet')*-1);
    bldiff = bedlevele-bedlevelb;
    
    
    SBU = squeeze(vs_let(trim,'map-sed-series',{1:124},'SBUU',{1:ycel 1:xcel 1:1},'quiet'));
    SBV = squeeze(vs_let(trim,'map-sed-series',{1:124},'SBVV',{1:ycel 1:xcel 1:1},'quiet'));
    SB = sqrt(SBU.^2+SBV.^2);       % = TOTAL TRANSPORT
    SB(SBU<0) = SB(SBU<0)*(-1);
    
    cumulative_transport = squeeze(sum(SB));
    cumulative_transport(1,:) = NaN;
    cumulative_transport(end,:) = NaN;
    cumulative_transport(:,1) = NaN;
    cumulative_transport(:,end) = NaN;
    cumulative_tr_cross = nansum(cumulative_transport); %cross-section
    
    
    if k < 4

        TrC = cumulative_tr_cross;
            TrmC = cumulative_transport;
blC = bldiff;
                    figure
        hold on
        title(models(k).name)
        surf(XZ,YZ,cumulative_transport,'EdgeColor','none');
        colormap(flipud(redblue))
        view([0 90])
        shading interp
        colorbar
        xlim([10 30])
        ylim([5 10])
        caxis([-1.5 1.5].*10^-3);box on;grid off;
        
% save figures
set(gcf,'PaperPositionMode','auto');
%  print(['/home/618467/netsedimentransport_',models(k).name],'-dpng','-r150');
 
    else
                    figure
        hold on
        title(models(k).name)
        surf(XZ,YZ,cumulative_transport,'EdgeColor','none');
        colormap(flipud(redblue))
        view([0 90])
        shading interp
        colorbar
        xlim([10 30])
        ylim([5 10])
        caxis([-1.5 1.5].*10^-3);box on;grid off;
        
% save figures
set(gcf,'PaperPositionMode','auto');
%  print(['/home/618467/netsedimentransport_',models(k).name],'-dpng','-r150');
        
%         figure
%         hold on
%         title(models(k).name)
%         surf(XZ,YZ,bldiff-blC,'EdgeColor','none');
%         colormap(flipud(redblue))
%         view([0 90])
%         shading interp
%         colorbar
%         xlim([10 30])
%         ylim([5 10])
%         caxis([-1 1].*10^-3);box on;grid off;
        
% save figures
set(gcf,'PaperPositionMode','auto');
%  print(['/home/618467/diff_netsedimentransport_',models(k).name],'-dpng','-r150');
        
        
        
    end

 
    
    
    

    
%     close all
    
    n = n+1;
 
end

%% legend

% saveas(gcf,['/home/618467/bathy_W'],'png')