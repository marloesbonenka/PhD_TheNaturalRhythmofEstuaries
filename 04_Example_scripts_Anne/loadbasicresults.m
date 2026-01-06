% plot basic results any Delft3D model
% Last edit: December 2024
% Author: Anne Baar

clear all; close all;clc;

%% paths (EDIT THIS)
addpath('C:\Users\awbaar\Documents\Delft3D models\matlab scripts\delft3dmatlab\') % add the location where you stored all the matlabscripts of deltares 

modellocation = (['C:\Users\awbaar\Documents\Delft3D models\Smaller estuary\dredging tests\M0 - run test\']); % add the folder with this script and the model result files (-trim or -trih))
cd(modellocation)

timestep = 12; % timestep you want to plot

%% load data
trim = vs_use([modellocation 'trim-estuary.dat'],[modellocation 'trim-estuary.def']); %loading the results file

XZ = squeeze(vs_let(trim,'map-const','XZ')); %loading the X coordinates of your grid
YZ = squeeze(vs_let(trim,'map-const','YZ')); %loading the Y coordinates of your grid
XZ(XZ == 0) = NaN;
YZ(YZ == 0) = NaN;

bedlevel = squeeze(vs_let(trim,'map-sed-series',{timestep:timestep},'DPS','quiet')*-1); %load the depth result file (note: multiplied with -1 to make it bedlevels instead of depth!)

waterlevel = squeeze(vs_let(trim,'map-series',{timestep:timestep},'S1','quiet')); %load the water levels


% to load other variables: type 'vs_let(trim)'. This opens a GUI with all the variables.
% you can look at the commandstructure for the bedlevel and waterlevel, compare it with the GUI, and see if you can load other variables this way!

%% plot bathymetry

figure; hold on % opens a new figure window

surf(XZ,YZ,bedlevel) % this plots a map (X,Y,Z coordinates)

view([0 90])
shading interp

xlim([nanmin(XZ(2,:)), nanmax(XZ(2,:))]) %axes limits
ylim([nanmin(YZ(:,2)), nanmax(YZ(:,2))])
xlabel('[m]') %axes labels
ylabel('[m]')
colorbar    %adds a colorbar to the plot

%% plot waterlevels

figure
hold on
surf(XZ,YZ,waterlevel)
view([0 90])
shading interp
xlim([nanmin(XZ(2,:)), nanmax(XZ(2,:))])
ylim([nanmin(YZ(:,2)), nanmax(YZ(:,2))])
xlabel('[m]')
ylabel('[m]')
colorbar
