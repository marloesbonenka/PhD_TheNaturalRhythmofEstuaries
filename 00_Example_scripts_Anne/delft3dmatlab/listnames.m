function objectname=listnames(handle,varargin)
%LISTNAMES Name graphics objects.
%
%   Names=LISTNAMES(Handles)
%   returns a cell array containing the names
%   of graphics objects

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
%   HeadURL: https://svn.oss.deltares.nl/repos/delft3d/trunk/src/tools_lgpl/matlab/quickplot/progsrc/private/listnames.m 
%   Id: listnames.m 2498 2013-05-16 12:38:27Z jagers 

handle=handle(:);
if isempty(handle)
    objectname={};
    return
elseif any(~isnumeric(handle)) | any(~ishandle(handle))
    objectname={};
    warning('Invalid handle passed to function LISTNAMES.')
    return
end
showType = 1;
showHandle = 1;
showTag = 1;
for i=1:2:length(varargin)
    switch lower(varargin{i})
        case 'showtype'
            showType = getbool(varargin{i+1});
        case 'showhandle'
            showHandle = getbool(varargin{i+1});
        case 'showtag'
            showTag = getbool(varargin{i+1});
    end
end
if ~isempty(handle)
    for i=1:length(handle)
        TagStr=get(handle(i),'tag');
        TypeStr=get(handle(i),'type');
        HandleStr=num2str(handle(i));
        switch TypeStr
            case 'figure'
                StringStr=get(handle(i),'name');
                if strcmp(get(handle(i),'numbertitle'),'on')
                    if matlabversionnumber < 7
                        NoString = ' No. ';
                    else
                        NoString = ' ';
                    end
                    if isempty(StringStr)
                        StringStr=[TypeStr NoString HandleStr];
                    else
                        StringStr=[TypeStr NoString HandleStr ':' StringStr];
                    end
                    StringStr(1)=upper(StringStr(1));
                end
            case 'axes'
                StringStr=get(get(handle(i),'title'),'string');
            case 'uicontrol'
                StringStr=get(handle(i),'string');
                if iscell(StringStr)
                    if isempty(StringStr)
                        StringStr='{}';
                    else
                        StringStr=['{''' StringStr{1} ''' ...}'];
                    end
                end
            case 'uimenu'
                StringStr=get(handle(i),'label');
            case 'text'
                parenthandle=get(handle(i),'parent');
                specobj=[get(parenthandle,'xlabel') get(parenthandle,'ylabel') get(parenthandle,'zlabel') get(parenthandle,'title')];
                specobjtype={'xlabel','ylabel','zlabel','title'};
                if ismember(handle(i),specobj) % label or title
                    TypeStr=specobjtype{min(find(specobj==handle(i)))};
                end
                StringStr=get(handle(i),'string');
            otherwise
                StringStr='';
        end
        if iscell(StringStr)
            if isempty(StringStr)
                StringStr='';
            else
                StringStr=StringStr{1};
            end
        end
        if ~isempty(StringStr)
            StringStr = StringStr(1,:);
        end
        if showType | showHandle | (showTag & ~isempty(TagStr))
            if ~isempty(StringStr)
                StringStr = [StringStr ' '];
            end
            StringStr = [StringStr '['];
        end
        if showType
            if strcmp(get(handle(i),'handlevisibility'),'off')
                TypeStr=['*' TypeStr];
            end
            StringStr = [StringStr TypeStr];
        end
        if showHandle
            if showType
                StringStr = [StringStr ' '];
            end
            StringStr = [StringStr HandleStr];
        end
        if showTag & ~isempty(TagStr)
            if showType | showHandle
                StringStr = [StringStr ': '];
            end
            StringStr = [StringStr TagStr];
        end
        if showType | showHandle | (showTag & ~isempty(TagStr))
            StringStr = [StringStr ']'];
        end
        objectname{i} = StringStr;
    end
else
    objectname={};
end


function B = getbool(X)
if ischar(X)
    switch X
        case {'no','false','n','f'}
            B = 0;
        case {'yes','true','y','t'}
            B = 1;
    end
else
    B = X;
end
