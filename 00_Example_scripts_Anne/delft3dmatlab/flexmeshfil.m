function varargout=flexmeshfil(FI,domain,field,cmd,varargin)
%FLEXMESHFIL QP support for various unstructured mesh files.
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
%   HeadURL: https://svn.oss.deltares.nl/repos/delft3d/trunk/src/tools_lgpl/matlab/quickplot/progsrc/private/flexmeshfil.m 
%   Id: flexmeshfil.m 2498 2013-05-16 12:38:27Z jagers 

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
        return;
    case 'times'
        varargout={readtim(FI,Props,varargin{:})};
        return
    case 'stations'
        varargout={{}};
        return
    case 'subfields'
        varargout={getsubfields(FI,Props,varargin{:})};
        return
    otherwise
        [XYRead,DataRead,DataInCell]=gridcelldata(cmd);
end

DimFlag=Props.DimFlag;

% initialize and read indices ...
idx={[] [] 0 [] []};
fidx=find(DimFlag);

idx(fidx(1:length(varargin)))=varargin;

if isfield(Props,'ElmLayer')
    Faces = FI.Faces(FI.ElmLyr==Props.ElmLayer,:);
else
    Faces = FI.Faces;
end
if ~isequal(idx{M_},0)
    Faces = Faces(idx{M_},:);
end
i = Faces>0;
%
switch Props.Geom
    case 'POLYG'
        X = Faces;
        Y = Faces;
        X(i) = FI.NodeCoor(Faces(i),1);
        Y(i) = FI.NodeCoor(Faces(i),2);
        X(~i) = NaN;
        Y(~i) = NaN;
        X(:,end+1:end+2) = NaN;
        Y(:,end+1:end+2) = NaN;
        for i=1:size(X,1)
            for j=1:size(X,2)
                if isnan(X(i,j))
                    X(i,j) = X(i,1);
                    Y(i,j) = Y(i,1);
                    break
                end
            end
        end
        X = X';
        Y = Y';
        Ans.X = X(:);
        Ans.Y = Y(:);
    case 'TRI'
        Ans.TRI = Faces;
        sz = size(FI.NodeCoor);
        Ans.XYZ = reshape(FI.NodeCoor,[1 sz(1) 1 sz(2)]);
        Ans.Val = FI.NodeCoor(:,3);
end
%
varargout={Ans FI};
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
function Out=infile(FI,domain)
T_=1; ST_=2; M_=3; N_=4; K_=5;
%======================== SPECIFIC CODE =======================================
PropNames={'Name'                   'Units' 'Geom' 'Coords' 'DimFlag' 'DataInCell' 'NVal' 'SubFld' 'ClosedPoly'};
DataProps={'mesh'                   ''     'POLYG' 'xy'    [0 0 6 0 0]  0            0      []         1
           'value'                  ''     'TRI'   'xy'    [0 0 6 0 0]  0            1      []         1};
if size(FI.NodeCoor,2)<3
    DataProps(2,:) = [];
end
Out=cell2struct(DataProps,PropNames,2);
if isfield(FI,'ElmLyr') && domain<=length(FI.Layers)
    [Out.ElmLayer] = deal(FI.Layers(domain));
end
% -----------------------------------------------------------------------------

% -----------------------------------------------------------------------------
function sz=getsize(FI,Props)
T_=1; ST_=2; M_=3; N_=4; K_=5;
sz=[0 0 0 0 0];
if isfield(Props,'ElmLayer')
    sz(M_) = sum(FI.ElmLyr==Props.ElmLayer);
else
    sz(M_) = size(FI.Faces,1);
end
% -----------------------------------------------------------------------------

% -----------------------------------------------------------------------------
function Domains=domains(FI)
if isfield(FI,'ElmLyr')
    nLyr = length(FI.Layers);
    Domains = cell(1,nLyr+1);
    for i = 1:nLyr
        Domains{i} = sprintf('%i',FI.Layers(i));
    end
    Domains{end} = 'All';
else
    Domains = {};
end
% -----------------------------------------------------------------------------
