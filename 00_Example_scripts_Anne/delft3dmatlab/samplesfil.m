function varargout=samplesfil(FI,domain,field,cmd,varargin)
%SAMPLESFIL QP support for XYZ sample files.
%   Domains                 = XXXFIL(FI,[],'domains')
%   DataProps               = XXXFIL(FI,Domain)
%   Size                    = XXXFIL(FI,Domain,DataFld,'size')
%   Times                   = XXXFIL(FI,Domain,DataFld,'times',T)
%   StNames                 = XXXFIL(FI,Domain,DataFld,'stations')
%   SubFields               = XXXFIL(FI,Domain,DataFld,'subfields')
%   [Data      ,NewFI]      = XXXFIL(FI,Domain,DataFld,'data',subf,t,station,m,n,k)
%   [Data      ,NewFI]      = XXXFIL(FI,Domain,DataFld,'celldata',subf,t,station,m,n,k)
%   [Data      ,NewFI]      = XXXFIL(FI,Domain,DataFld,'griddata',subf,t,station,m,n,k)
%   [Data      ,NewFI]      = XXXFIL(FI,Domain,DataFld,'gridcelldata',subf,t,station,m,n,k)
%                             XXXFIL(FI,[],'options',OptionsFigure,'initialize')
%   [NewFI     ,cmdargs]    = XXXFIL(FI,[],'options',OptionsFigure,OptionsCommand, ...)
%
%   The DataFld can only be either an element of the DataProps structure.

%----- LGPL --------------------------------------------------------------------
%                                                                               
%   Copyright (C) 2011-2013 Stichting Deltares.                                     
%                                                                               
%   This library is free software; you can redistribute it and/or                
%   modify it under the terms of the GNU Lesser General Public                   
%   License as published by the Free Software Foundation version 2.1.                         
%                                                                               
%   This library is distributed in the hope that it will be useful,              
%   but WITHOUT ANY WARRANTY; without even the implied warranty of               
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU            
%   Lesser General Public License for more details.                              
%                                                                               
%   You should have received a copy of the GNU Lesser General Public             
%   License along with this library; if not, see <http://www.gnu.org/licenses/>. 
%                                                                               
%   contact: delft3d.support@deltares.nl                                         
%   Stichting Deltares                                                           
%   P.O. Box 177                                                                 
%   2600 MH Delft, The Netherlands                                               
%                                                                               
%   All indications and logos of, and references to, "Delft3D" and "Deltares"    
%   are registered trademarks of Stichting Deltares, and remain the property of  
%   Stichting Deltares. All rights reserved.                                     
%                                                                               
%-------------------------------------------------------------------------------
%   http://www.deltaressystems.com
%   HeadURL: https://svn.oss.deltares.nl/repos/delft3d/trunk/src/tools_lgpl/matlab/quickplot/progsrc/private/samplesfil.m 
%   Id: samplesfil.m 2699 2013-07-11 08:02:51Z jagers 

%========================= GENERAL CODE =======================================

T_=1; ST_=2; M_=3; N_=4; K_=5;

if nargin<2
    error('Not enough input arguments');
elseif nargin==2
    varargout={infile(FI,domain)};
    return
elseif ischar(field)
    switch field
        case 'options'
            [varargout{1:2}]=options(FI,cmd,varargin{:});
        case 'optionstransfer'
            varargout{1}=optionstransfer(FI,cmd);
        case 'domains'
            varargout={domains(FI)};
        case 'dimensions'
            varargout={dimensions(FI)};
        case 'locations'
            varargout={locations(FI)};
        case 'quantities'
            varargout={quantities(FI)};
        case 'data'
            [varargout{1:2}]=getdata(FI,cmd,varargin{:});
    end
    return
else
    Props=field;
end

cmd=lower(cmd);
switch cmd
    case 'size'
        varargout={getsize(FI,Props)};
        return
    case 'times'
        varargout={readtim(FI,Props,varargin{:})};
        return
    case 'stations'
        varargout={readsts(FI,Props,varargin{:})};
        return
    case 'subfields'
        varargout={{}};
        return
    otherwise
        [XYRead,DataRead,DataInCell]=gridcelldata(cmd);
end

DimFlag=Props.DimFlag;

% initialize and read indices ...
idx={[] [] 0 0 0};
fidx=find(DimFlag);
idx(fidx(1:length(varargin)))=varargin;

% select appropriate timestep ...
sz=getsize(FI,Props);
if DimFlag(T_)
    if isempty(idx{T_})
        idx{T_}=sz(T_);
    end
    if isequal(idx{T_},0)
        idx{T_}=1:sz(T_);
    end
elseif ~isempty(FI.Time)
    idx{T_} = 1;
end
nTim = max(1,length(idx{T_}));
nLoc = max([sz(M_) sz(ST_)]);
if DimFlag(ST_)
    idxM = idx{ST_};
    szM = sz(ST_);
else
    idxM = idx{M_};
    szM = sz(M_);
end
if isempty(idx{T_})
    if isequal(idxM,0)
        dim1 = ':';
    else
        dim1 = idxM;
        nLoc = length(idxM);
    end
else
    dim1 = ismember(FI.iTime,idx{T_});
    %
    if ~isequal(idxM,0) % can only happen if number of locations if constant, i.e. FI.nLoc scalar
        z = zeros(szM,1);
        z(idxM) = 1;
        dim1 = dim1 & repmat(z,[sz(T_) 1]);
        nLoc = length(idxM);
    end
end

% generate output ...
if XYRead
    xyz = FI.XYZ(dim1,[FI.X FI.Y]);
    nPnt=size(xyz,1)/nTim;
    nCrd=size(xyz,2);
    Ans.XYZ=reshape(xyz,[nTim nPnt 1 nCrd]);
    if strcmp(Props.Geom,'TRI')
       if isfield(FI,'TRI')
          Ans.TRI=FI.TRI;
       elseif ~isempty(FI.X) && ~isempty(FI.Y)
          try
             [xy,I]=unique(xyz,'rows');
             tri=delaunay(xy(:,1),xy(:,2));
             Ans.TRI=I(tri);
             if length(FI.nLoc)==1
                 FI.TRI=Ans.TRI;
             end
          catch
             Ans.TRI=zeros(0,3);
          end
       end
    else
       Ans.TRI=zeros(0,3);
    end
end

switch Props.NVal
    case 0
    case 1
        if nLoc==0 % if variable number of locations, then nTim==1
            Ans.Val=FI.XYZ(dim1,Props.SubFld)';
        else
            Ans.Val=reshape(FI.XYZ(dim1,Props.SubFld),[nTim nLoc]);
        end
    otherwise
        Ans.XComp=[];
        Ans.YComp=[];
end

% read time ...
T=readtim(FI,Props,idx{T_});
Ans.Time=T;

varargout={Ans FI};
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
function Out=infile(FI,domain)

PropNames={'Name'                       'Units' 'DimFlag' 'DataInCell' 'NVal' 'VecType' 'Loc' 'ReqLoc' 'Geom' 'Coords' 'SubFld'};
DataProps={'locations'                  ''       [0 0 1 0 0]  0          0     ''        ''    ''     'PNT'  'xy'      []
  'triangulated locations'              ''       [0 0 1 0 0]  0          0     ''        ''    ''     'TRI'  'xy'      []
  '-------'                             ''       [0 0 0 0 0]  0          0     ''        ''    ''     ''     ''        []
  'sample data'                         ''       [0 0 1 0 0]  0          1     ''        ''    ''     'TRI'  'xy'      -999};

Out=cell2struct(DataProps,PropNames,2);

params = 1:size(FI.XYZ,2);
params = setdiff(params,[FI.X FI.Y FI.Time]);
if ~isempty(FI.Time)
    if length(FI.nLoc)==1
        f3 = 1;
    else
        f3 = inf; % variable number of nodes
    end
    %
    Out(end).DimFlag(1) = 1;
    for i = [1 2 4]
        Out(i).DimFlag(3) = f3;
    end
end

% Expand parameters
NPar=length(params);
if NPar>0
    Out=cat(1,Out(1:3),repmat(Out(4),NPar,1));
    for i = 1:NPar
        Out(i+3).SubFld = params(i);
        Out(i+3).Name   = FI.Params{params(i)};
        if isfield(FI,'ParamUnits')
            Out(i+3).Units  = FI.ParamUnits{params(i)};
        end
    end
else
   Out=Out(1:2);
end

% No triangulation possible if only one or two points, or only one
% coordinate
if (length(FI.nLoc)==1 && FI.nLoc<2) || isempty(FI.Y) || isempty(FI.X)
   Out(2)=[];
   for i=1:NPar
      Out(i+2).Geom='PNT';
      if isempty(FI.Y)
          Out(i+2).Coords='x';
      elseif isempty(FI.X)
          Out(i+2).Coords='y';
      end
   end
end

if isempty(FI.Y) || isempty(FI.X)
    for i=1:NPar
        Out(i+2).DimFlag(2) = 5;
        Out(i+2).DimFlag(3) = 0;
    end
    %
    Out(1:2) = [];
end
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
function sz=getsize(FI,Props)
T_=1; ST_=2; M_=3; N_=4; K_=5;
sz=[0 0 0 0 0];
%======================== SPECIFIC CODE =======================================
if Props.DimFlag(T_)
    sz(T_) = length(FI.Times);
end
if Props.DimFlag(M_) && length(FI.nLoc)==1
    sz(M_) = FI.nLoc;
elseif Props.DimFlag(ST_)
    sz(ST_) = FI.nLoc;
end
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
function T=readtim(FI,Props,t)
%======================== SPECIFIC CODE =======================================
if isempty(FI.Time)
    T = 0;
else
    T=FI.Times;
    if t~=0
        T = T(t);
    end
end
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
function S=readsts(FI,Props,t)
%======================== SPECIFIC CODE =======================================
if nargin<3 || t==0
    S = cell(1,FI.nLoc);
    t=1:FI.nLoc;
else
    S = cell(1,length(t));
end
for i=1:length(t)
    XUnit = '';
    YUnit = '';
    if isfield(FI,'ParamUnits')
        if ~isempty(FI.X)
            XUnit = [' ' FI.ParamUnits{FI.X}];
        end
        if ~isempty(FI.Y)
            YUnit = [' ' FI.ParamUnits{FI.Y}];
        end
    end
    %
    if ~isempty(FI.X) && isempty(FI.Y)
        S{i} = sprintf('x = %g%s',FI.XYZ(t(i),FI.X),XUnit);
    elseif isempty(FI.X) && ~isempty(FI.Y)
        S{i} = sprintf('y = %g%s',FI.XYZ(t(i),FI.Y),YUnit);
    else
        S{i} = sprintf('(x,y) = (%g%s,%g%s)',FI.XYZ(t(i),FI.X),XUnit,FI.XYZ(t(i),FI.Y),YUnit);
    end
end
% -----------------------------------------------------------------------------
