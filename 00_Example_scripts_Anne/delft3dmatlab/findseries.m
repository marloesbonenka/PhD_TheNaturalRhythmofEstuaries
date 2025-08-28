function bs=findseries(bool);
%FINDSERIES  Find series of nonzero elements.
%   I=FINDSERIES(X) returns an Nx2 array of
%   pairs of indices referring to the start
%   and end of series of nonzero elements.
%   First column of I: start index.
%   Second column of I: end index.

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
%   HeadURL: https://svn.oss.deltares.nl/repos/delft3d/trunk/src/tools_lgpl/matlab/quickplot/progsrc/private/findseries.m 
%   Id: findseries.m 2498 2013-05-16 12:38:27Z jagers 

ibool=find(bool(:));
if isempty(ibool), bs=zeros(0,2); return; end
istart=ibool(cat(2,1,find(diff(ibool')>1)+1));
iend=ibool(cat(2,find(diff(ibool')>1),length(ibool)));
bs=[istart iend];
