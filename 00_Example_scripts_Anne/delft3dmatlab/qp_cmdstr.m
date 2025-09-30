function [cmd,cmdargs]=qp_cmdstr(cmdstr)
%QP_CMDSTR Process QuickPlot command string.

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
%   HeadURL: https://svn.oss.deltares.nl/repos/delft3d/trunk/src/tools_lgpl/matlab/quickplot/progsrc/private/qp_cmdstr.m 
%   Id: qp_cmdstr.m 2498 2013-05-16 12:38:27Z jagers 

cmdargs={};
[cmd,rmndr]=strtok(cmdstr);
cmd=lower(cmd);
while ~isempty(rmndr)
    idx=find(rmndr~=32);
    if isempty(idx)
        rmndr='';
    else
        switch rmndr(idx(1))
            case ''''
                rmndr=rmndr(idx(1)+1:end);
                quotes=find(rmndr=='''');
                i=1; endquote=[];
                while i<=length(quotes) & isempty(endquote)
                    if i==length(quotes) | quotes(i+1)~=quotes(i)+1
                        endquote=quotes(i);
                    else
                        i=i+2;
                    end
                end
                if isempty(endquote)
                    error('Invalid command argument encountered: ''%s',rmndr)
                else
                    cmdargs{end+1}=rmndr(1:endquote-1);
                    rmndr=rmndr(endquote+1:end);
                end
            case '['
                rmndr=rmndr(idx(1)+1:end);
                [X,count,err,n]=sscanf(rmndr,'%[0-9+ .Ee:-]');
                if n>length(rmndr) | rmndr(n)~=']'
                    error('Invalid command argument encountered: [%s',rmndr)
                end
                cmdargs{end+1}=X;

                rmndr=rmndr(n+1:end);
            case {'1','2','3','4','5','6','7','8','9','0','.','-','+'}
                [X,count,err,n]=sscanf(rmndr,'%f',1);
                if count==1
                    cmdargs{end+1}=X;
                    rmndr=rmndr(n:end);
                else
                    error(err);
                end
            otherwise
                error('Invalid command argument encountered: %s',rmndr)
        end
    end
end
