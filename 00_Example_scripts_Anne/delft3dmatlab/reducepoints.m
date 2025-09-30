function varargout = reducepoints(varargin);
%REDUCEPOINTS Filters a set points using a distance threshold.
%   I = REDUCEPOINTS(Thresh_Dist,X,Y,Z)
%   returns an array I of indices of points that form
%   together a set of points that are mutually separated
%   by at least a distance Thresh_Dist. The function
%   works in 1 (X), 2 (X and Y) and 3 (X, Y and Z) dimensions.
%
%   See also REDUCEPNTSQ.

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
%   HeadURL: https://svn.oss.deltares.nl/repos/delft3d/trunk/src/tools_lgpl/matlab/quickplot/progsrc/private/reducepoints.m 
%   Id: reducepoints.m 2498 2013-05-16 12:38:27Z jagers 

%#function reducepoints_r13_6p5
%#function reducepoints_r2007a_7p4

v = matlabversionnumber;
if v>=7.04
    fcn='reducepoints_r2007a_7p4';
elseif v>=6.05
    fcn='reducepoints_r13_6p5';
else
    error('This MATLAB version is too old.');
end
[varargout{1:max(1,nargout)}]=calldll(fcn,varargin{:});

