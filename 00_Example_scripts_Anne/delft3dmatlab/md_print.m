function [Settings,fn]=md_print(varargin)
%MD_PRINT Send a figure to a printer.
%   MD_PRINT(FigureHandles)
%   Opens the user interface for the specified figures (in the
%   specified order). If no argument is specified, the command works
%   on the current figure (if it exists).
%
%   Settings=MD_PRINT(...)
%   Returns the (last) selected output mode and the settings. To use
%   the UI for selection only (i.e. no actual printing/saving of a
%   figure), use an empty list of figures: Settings=MD_PRINT([])
%
%   MD_PRINT(FigureHandles,Settings,FileName)
%   Prints the specified figures using the specified settings.
%   The filename is optional for output to file (PS/EPS/TIF/JPG/EMF/PNG).
%
%   See also PRINT.

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
%   HeadURL: https://svn.oss.deltares.nl/repos/delft3d/trunk/src/tools_lgpl/matlab/quickplot/progsrc/private/md_print.m 
%   Id: md_print.m 3085 2013-11-02 16:55:24Z jagers 

%  Painters   'Printer Name'         Platforms    COLOR    MultiPage
%  COLOR=0 Never, COLOR=1 User Specified, COLOR=2 Always

% Called by print -dsetup:
%#function orient

persistent PL

if isempty(PL)
    W  = 1;
    WD = 2;
    U  = 4;
    UD = 8;
    all = W+WD+U+UD;
    win = W+WD;
    unx = U+UD;
    %
    PL={1  'PDF file'                    all        1     1
        1  'PS file'                     all        1     0
        1  'EPS file'                    all        1     0
        0  'TIF file'                    all        2     0
        0  'BMP file'                    all        2     0
        0  'PNG file'                    all        2     0
        0  'JPG file'                    all        2     0
        1  'EMF file'                    win        2     0
        -1  'MATLAB fig file'            all        2     0
        0  'Bitmap to clipboard'         win        2     0
        1  'Metafile to clipboard'       win        2     0
        1  'Windows printer'             W          2     0
        1  'default Windows printer'     WD         2     0
        1  'other Windows printer'       WD         2     0};
    if isdeployed
        if ispc
            code = WD;
        else
            code = UD;
        end
    else
        if ispc
            code = W;
        else
            code = U;
        end
    end
    Remove = ~bitget(cat(1,PL{:,3}),log2(code)+1);
    PL(Remove,:)=[];
end

getsettings = 0;
nArgParsed = 0;
Local_ui_args = {};
if nargin==0
    shh=get(0,'showhiddenhandles');
    set(0,'showhiddenhandles','on');
    figlist=get(0,'currentfigure');
    Local_ui_args={get(0,'children') figlist};
    set(0,'showhiddenhandles',shh);
else
    figlist = varargin{1};
    nArgParsed = 1;
    if isequal(figlist,'getsettings')
        getsettings = 1;
        figlist = varargin{2};
        nArgParsed = 2;
    elseif ischar(figlist)
        %
        % Deal with uicontrol callbacks ...
        %
        UD=get(gcbf,'userdata');
        UD{end+1}=figlist;
        set(gcbf,'userdata',UD);
        return
    end
end

selfiglist=intersect(figlist,allchild(0));
if ~isequal(sort(figlist),selfiglist) && ~isempty(figlist)
    fprintf('Non-figure handles removed from print list.\n');
    figlist=selfiglist;
end

if nargin<nArgParsed+1
    LocSettings.PrtID=-1;
    LocSettings.AllFigures=0;
else
    LocSettings = varargin{nArgParsed+1};
    nArgParsed = nArgParsed+1;
end
if ischar(LocSettings.PrtID)
    LocSettings.PrtID=ustrcmpi(LocSettings.PrtID,PL(:,2));
elseif LocSettings.PrtID>=0
    error('Please replace PrtID number by printer name.');
end

if isfield(LocSettings,'SelectFrom')
    Local_ui_args={LocSettings.SelectFrom figlist};
end

if (isempty(figlist) && nargout>0) || getsettings
    if nargout>0
        [Settings,figlist]=Local_ui(PL,0,LocSettings,Local_ui_args{:});
        Settings.AllFigures=1;
        if Settings.PrtID>0
            Settings.PrtID=PL{Settings.PrtID,2};
        end
        fn=figlist;
    end
    return
end

if nargin>=nArgParsed+1
    fn=varargin{nArgParsed+1};
else
    fn='';
end

i=0;
tempfil='';
pagenr = 1;
while i<length(figlist)
    i=i+1;
    if ishandle(figlist(i))
        if strcmp(get(figlist(i),'type'),'figure')
            if LocSettings.PrtID<0
                figure(figlist(i));
                if i>1
                    Local_ui_args = {};
                end
                [LocSettings,figlistnew]=Local_ui(PL,(length(figlist)-i)>0,[],Local_ui_args{:});
                if ~isempty(Local_ui_args)
                    LocSettings.AllFigures = 1;
                    figlist = figlistnew;
                    i = 1;
                end
                tempfil=[tempname,'.'];
                MoreToCome=LocSettings.AllFigures;
            else % LocSettings.AllFigures continued
                MoreToCome=i<length(figlist);
            end
            if isempty(tempfil)
                tempfil=[tempname,'.'];
            end

            hvis=get(figlist(i),'handlevisibility');
            set(figlist(i),'handlevisibility','on');
            % figure(figlist(i));
            FigStr=sprintf('-f%20.16f',figlist(i));

            if LocSettings.PrtID==0
                Printer='cancel';
            else
                Printer=PL{LocSettings.PrtID,2};
                switch LocSettings.Method
                    case 1 % Painters
                        PrtMth={'-painters'};
                    case 2 % Zbuffer
                        PrtMth={'-zbuffer',sprintf('-r%i',LocSettings.DPI)};
                end
            end
            switch Printer
                case 'cancel'
                    % nothing to do
                case {'TIF file','BMP file','PNG file','JPG file','EPS file','PS file','EMF file','PDF file'}
                    switch Printer
                        case 'TIF file'
                            ext='tif';
                            dvr='-dtiff';
                        case 'BMP file'
                            ext='bmp';
                            dvr='-dbmp';
                        case 'PNG file'
                            ext='png';
                            dvr='-dpng';
                        case 'JPG file'
                            ext='jpg';
                            dvr='-djpeg';
                        case 'PDF file'
                            ext='pdf';
                            dvr='-dps';
                            if LocSettings.Color
                                dvr='-dpsc';
                            end
                        case 'EPS file'
                            ext='eps';
                            dvr='-deps';
                            if LocSettings.Color
                                dvr='-depsc';
                            end
                        case 'PS file'
                            ext='ps';
                            dvr='-dps';
                            if LocSettings.Color
                                dvr='-dpsc';
                            end
                        case 'EMF file'
                            ext='emf';
                            dvr='-dmeta';
                    end
                    if nargin<3 && pagenr==1
                        [fn,pn]=uiputfile(['default.' ext],'Specify file name');
                        fn = [pn,fn];
                    end
                    if ischar(fn)
                        ih=get(figlist(i),'inverthardcopy');
                        if isfield(LocSettings,'InvertHardcopy') && ~LocSettings.InvertHardcopy
                            set(figlist(i),'inverthardcopy','off');
                        else
                            LocSettings.InvertHardcopy=1;
                            set(figlist(i),'inverthardcopy','on');
                        end
                        if strcmp(Printer,'PDF file')
                            switch pagenr
                                case 1
                                    pdfname = fn;
                                    pagenr = 1;
                                    fn = [tempname '.ps'];
                                otherwise
                                    PrtMth{end+1}='-append';
                            end
                        end
                        normtext = findall(figlist(i),'fontunits','normalized');
                        if ~isempty(normtext)
                            set(normtext,'fontunits','points')
                            drawnow % needed to get output file nicely formatted
                        end
                        try
                            print(fn,FigStr,dvr,PrtMth{:});
                            if strcmp(Printer,'PDF file')
                                figname = getappdata(figlist(i),'md_print_name');
                                if isempty(figname)
                                    figname = sprintf('page %i',pagenr);
                                end
                                add_bookmark(fn, figname,pagenr==1)
                                if ~MoreToCome
                                    ps2pdf('psfile',fn,'pdffile',pdfname,'gspapersize','a4','deletepsfile',1)
                                    fn = pdfname;
                                end
                                pagenr = pagenr+1;
                            end
                        catch
                            ui_message('error','error encountered creating %s:%s',fn,lasterr);
                        end
                        if ~isempty(normtext)
                            set(normtext,'fontunits','normalized')
                        end
                        set(figlist(i),'inverthardcopy',ih);
                    end
                case 'MATLAB fig file'
                    if nargin<3
                        [fn,pn]=uiputfile('default.fig','Specify file name');
                        fn=[pn,fn];
                    end
                    if ischar(fn)
                        try
                            hgsave(figlist(i),fn);
                        catch
                            ui_message('error','error encountered creating %s:%s',fn,lasterr);
                        end
                    end
                otherwise
                    try
                        ccd=cd;
                        cd(tempdir);
                        ih=get(figlist(i),'inverthardcopy');
                        if isfield(LocSettings,'InvertHardcopy') && ~LocSettings.InvertHardcopy
                            set(figlist(i),'inverthardcopy','off');
                        else
                            LocSettings.InvertHardcopy=1;
                            set(figlist(i),'inverthardcopy','on');
                        end
                        switch Printer
                            case {'other Windows printer','default Windows printer','Windows printer'}
                                paperpos=get(figlist(i),'paperposition');
                                %set(figlist(i),'paperposition',paperpos-[0.5 0 0.5 0]);
                                if isdeployed && strcmp(Printer,'default Windows printer')
                                    deployprint(figlist(i))
                                elseif isdeployed && strcmp(Printer,'other Windows printer')
                                    printdlg(figlist(i))
                                else
                                    dvr='-dwin';
                                    if LocSettings.Color
                                        dvr='-dwinc';
                                    end
                                    print(FigStr,dvr,PrtMth{:});
                                end
                                set(figlist(i),'paperposition',paperpos);
                            case 'Bitmap to clipboard'
                                set(figlist(i),'inverthardcopy','off');
                                print(FigStr,'-dbitmap');
                            case 'Metafile to clipboard'
                                print(FigStr,PrtMth{:},'-dmeta');
                        end
                        set(figlist(i),'inverthardcopy',ih);
                        cd(ccd);
                    catch
                        ui_message('error','error encountered printing to %s:%s',Printer,lasterr);
                    end
            end
            if nargout>0
                Settings=LocSettings;
                Settings.AllFigures=1;
                if Settings.PrtID>0
                    Settings.PrtID=PL{Settings.PrtID,2};
                end
            end
            set(figlist(i),'handlevisibility',hvis);
            if ~LocSettings.AllFigures,
                LocSettings.PrtID=-1;
            end
        end
    end
end

function [Settings,FigID]=Local_ui(PL,CanApplyAll,Settings,SelectFrom,FigID)
persistent PrtID Method DPI ApplyAll Clr InvertHardcopy
if isempty(PrtID)
    PrtID=1;
    Method=2;
    DPI=150;
    ApplyAll=0;
    InvertHardcopy=1;
    Clr=1;
end
if nargin>2 && isstruct(Settings)
    if isfield(Settings,'PrtID')
        PrtID=Settings.PrtID;
        if PrtID<=0
            PrtID=1;
        end
    end
    if isfield(Settings,'Method')
        Method=Settings.Method;
    end
    if isfield(Settings,'DPI')
        DPI=Settings.DPI;
    end
    if isfield(Settings,'ApplyAll')
        ApplyAll=Settings.ApplyAll;
    end
    if isfield(Settings,'InvertHardcopy')
        InvertHardcopy=Settings.InvertHardcopy;
    end
    if isfield(Settings,'Clr')
        Clr=Settings.Clr;
    end
end

Reselect = 0;
if nargin>3 && ~isempty(SelectFrom)
    Reselect = 1;
end
if nargin<5
    FigID = [];
end
XX=xx_constants;

PrintLabel=200;
TextLabel=50;
FigListHeight=100;

ZBufWidth=80;
ResWidth=50;

TextShift = [0 XX.Txt.Height-XX.But.Height 0 0];
Fig_Width=TextLabel+PrintLabel+3*XX.Margin;
Fig_Height=6*XX.Margin+7*XX.But.Height+XX.Txt.Height+ ...
    (XX.Margin+FigListHeight)*Reselect + ...
    (XX.Margin+XX.But.Height)*double(CanApplyAll);

ss = qp_getscreen;
swidth = ss(3);
sheight = ss(4);
left = ss(1)+(swidth-Fig_Width)/2;
bottom = ss(2)+(sheight-Fig_Height)/2;
rect = [left bottom Fig_Width Fig_Height];

fig=qp_uifigure('Print/Export','','md_print',rect);
set(fig,'closerequestfcn','closereq')

rect = [XX.Margin XX.Margin (Fig_Width-3*XX.Margin)/2 XX.But.Height];
uicontrol('style','pushbutton', ...
    'position',rect, ...
    'string','Cancel', ...
    'parent',fig, ...
    'callback','md_print cancel');

rect(1) = (Fig_Width+XX.Margin)/2;
OK=uicontrol('style','pushbutton', ...
    'position',rect, ...
    'string','OK', ...
    'parent',fig, ...
    'callback','md_print OK');

rect(1) = XX.Margin;
if CanApplyAll
    rect(2) = rect(2)+rect(4)+XX.Margin;
    rect(3) = Fig_Width-2*XX.Margin;
    rect(4) = XX.But.Height;
    AllFig=uicontrol('style','checkbox', ...
        'position',rect, ...
        'parent',fig, ...
        'value',ApplyAll, ...
        'string','Apply to All Remaining Figures', ...
        'backgroundcolor',XX.Clr.LightGray, ...
        'enable','on', ...
        'callback','md_print ApplyAll');
end

rect(2) = rect(2)+rect(4)+XX.Margin;
rect(3) = Fig_Width-2*XX.Margin;
rect(4) = XX.But.Height;
InvHard=uicontrol('style','checkbox', ...
    'position',rect, ...
    'parent',fig, ...
    'value',InvertHardcopy==1, ...
    'string','White Background and Black Axes', ...
    'backgroundcolor',XX.Clr.LightGray, ...
    'enable','on', ...
    'callback','md_print InvertHardcopy');

rect(2) = rect(2)+rect(4);
rect(3) = Fig_Width-2*XX.Margin;
rect(4) = XX.But.Height;
Color=uicontrol('style','checkbox', ...
    'position',rect, ...
    'parent',fig, ...
    'string','Print as Colour', ...
    'horizontalalignment','left', ...
    'value',Clr, ...
    'enable','on', ...
    'callback','md_print Color');

rect(1) = XX.Margin;
rect(2) = rect(2)+rect(4)+XX.Margin;
rect(3) = ZBufWidth;
rect(4) = XX.But.Height;
ZBuf=uicontrol('style','radiobutton', ...
    'position',rect, ...
    'parent',fig, ...
    'string','ZBuffer', ...
    'value',Method==2, ...
    'backgroundcolor',XX.Clr.LightGray, ...
    'enable','on', ...
    'callback','md_print Zbuffer');

rect(1) = rect(1)+rect(3)+XX.Margin;
rect(3) = ResWidth;
Resol=uicontrol('style','edit', ...
    'position',rect, ...
    'parent',fig, ...
    'string',num2str(DPI), ...
    'horizontalalignment','right', ...
    'backgroundcolor',XX.Clr.LightGray, ...
    'enable','off', ...
    'callback','md_print DPI');
if Method==2
    set(Resol,'enable','on','backgroundcolor',XX.Clr.White)
end

rect(1) = rect(1)+rect(3)+XX.Margin;
rect(3) = Fig_Width-(4*XX.Margin+ZBufWidth+ResWidth);
rect(4) = XX.Txt.Height;
uicontrol('style','text', ...
    'position',rect+TextShift, ...
    'parent',fig, ...
    'string','DPI', ...
    'horizontalalignment','left', ...
    'backgroundcolor',XX.Clr.LightGray, ...
    'enable','on');
rect(4) = XX.But.Height;

rect(1) = XX.Margin;
rect(2) = rect(2)+rect(4);
rect(3) = Fig_Width-2*XX.Margin;
rect(4) = XX.But.Height;
Painter=uicontrol('style','radiobutton', ...
    'position',rect, ...
    'parent',fig, ...
    'value',Method==1, ...
    'string','Painters (864 DPI)', ...
    'backgroundcolor',XX.Clr.LightGray, ...
    'enable','on', ...
    'callback','md_print Painters');

rect(1) = XX.Margin;
rect(2) = rect(2)+rect(4);
rect(3) = (Fig_Width-3*XX.Margin)/2;
rect(4) = XX.Txt.Height;
uicontrol('style','text', ...
    'position',rect+TextShift, ...
    'parent',fig, ...
    'string','Printing Method', ...
    'horizontalalignment','left', ...
    'backgroundcolor',XX.Clr.LightGray, ...
    'enable','on');

rect(1) = 2*XX.Margin+TextLabel;
rect(2) = rect(2)+rect(4)+XX.Margin;
rect(3) = PrintLabel/2;
rect(4) = XX.But.Height;
Opt=uicontrol('style','pushbutton', ...
    'position',rect, ...
    'parent',fig, ...
    'string','Options...', ...
    'horizontalalignment','left', ...
    'backgroundcolor',XX.Clr.LightGray, ...
    'enable','off', ...
    'callback','md_print Options');

rect(1) = XX.Margin;
rect(2) = rect(2)+rect(4)+XX.Margin;
rect(3) = TextLabel;
rect(4) = XX.Txt.Height;
uicontrol('style','text', ...
    'position',rect+TextShift, ...
    'parent',fig, ...
    'string','Printer', ...
    'horizontalalignment','left', ...
    'backgroundcolor',XX.Clr.LightGray, ...
    'enable','on');

rect(1) = 2*XX.Margin+TextLabel;
rect(3) = PrintLabel;
rect(4) = XX.But.Height;
Printer=uicontrol('style','popupmenu', ...
    'position',rect, ...
    'parent',fig, ...
    'string',PL(:,2), ...
    'value',PrtID, ...
    'horizontalalignment','left', ...
    'backgroundcolor',XX.Clr.White, ...
    'enable','on', ....
    'callback','md_print Printer');

if Reselect
    rect(1) = 2*XX.Margin+TextLabel;
    rect(2) = rect(2)+rect(4)+XX.Margin;
    rect(3) = PrintLabel;
    rect(4) = FigListHeight;
    AllFigID = SelectFrom;
    AllFigNames = listnames(AllFigID,'showType','no','showHandle','no','showTag','no');
    [AllFigNames,Order] = sort(AllFigNames);
    AllFigID = AllFigID(Order);
    FigIndex = find(ismember(AllFigID,FigID));
    FigLst=uicontrol('style','listbox', ...
        'position',rect, ...
        'parent',fig, ...
        'string',AllFigNames, ...
        'value',FigIndex, ...
        'max',2, ...
        'horizontalalignment','left', ...
        'backgroundcolor',XX.Clr.White, ...
        'enable','on', ....
        'callback','md_print Figure');

    rect(1) = XX.Margin;
    rect(2) = rect(2)+rect(4)-XX.Txt.Height;
    rect(3) = TextLabel;
    rect(4) = XX.Txt.Height;
    uicontrol('style','text', ...
        'position',rect+TextShift, ...
        'parent',fig, ...
        'string','Figure(s)', ...
        'horizontalalignment','left', ...
        'backgroundcolor',XX.Clr.LightGray, ...
        'enable','on');
end

set(fig,'visible','on','color',XX.Clr.LightGray);
set(fig,'userdata',{});

if strcmp(PL{PrtID,2},'Windows printer')
    set(Opt,'enable','on');
else
    set(Opt,'enable','off');
end
switch PL{PrtID,1}
    case -1 % painters/zbuffer irrelevant
        set(Painter,'value',1,'enable','off');
        set(ZBuf,'value',0,'enable','off');
        set(Resol,'backgroundcolor',XX.Clr.LightGray,'enable','off');
        Method=1;
    case 0 % cannot be combined with painters, e.g. bitmap
        set(Painter,'value',0,'enable','off');
        set(ZBuf,'value',1,'enable','on');
        set(Resol,'backgroundcolor',XX.Clr.White,'string',num2str(DPI),'enable','on');
        Method=2;
    otherwise
        set(Painter,'enable','on');
        set(ZBuf,'enable','on');
end

gui_quit=0;               % Becomes one if the interface has to quit.
stack=[];                 % Contains the stack of commands; read from 'userdata' field of the figure
Cancel=0;

while ~gui_quit

    %%*************************************************************************************************
    %%%% UPDATE SCREEN BEFORE WAITING FOR COMMAND
    %%*************************************************************************************************

    drawnow;

    %%*************************************************************************************************
    %%%% WAIT UNTIL A COMMAND IS ON THE STACK IN THE USERDATA FIELD OF THE FIGURE
    %%*************************************************************************************************

    if ishandle(fig)
        UD=get(fig,'userdata');
        if isempty(UD)
            waitfor(fig,'userdata');
        end
    end

    %%*************************************************************************************************
    %%%% SET POINTER TO WATCH WHILE PROCESSING COMMANDS ON STACK
    %%%% FIRST CHECK WHETHER FIGURE STILL EXISTS
    %%*************************************************************************************************

    if ishandle(fig)
        stack=get(fig,'userdata');
        set(fig,'userdata',{});
    else
        Cancel=1;
        gui_quit=1;
    end

    %%*************************************************************************************************
    %%%% START OF WHILE COMMANDS ON STACK LOOP
    %%*************************************************************************************************

    while ~isempty(stack)
        cmd=stack{1};
        stack=stack(2:size(stack,1),:);
        switch cmd,
            case 'cancel'
                Cancel=1;
                gui_quit=1;
            case 'OK'
                gui_quit=1;
            case 'Figure'
                FigIndex=get(FigLst,'value');
                FigID=AllFigID(FigIndex);
                if isempty(FigID)
                    set(OK,'enable','off')
                else
                    set(OK,'enable','on')
                end
            case 'Printer'
                PrtID=get(Printer,'value');
                if strcmp(PL{PrtID,2},'Windows printer')
                    set(Opt,'enable','on');
                else
                    set(Opt,'enable','off');
                end
                switch PL{PrtID,1}
                    case -1 % painters/zbuffer irrelevant
                        set(Painter,'value',1,'enable','off');
                        set(ZBuf,'value',0,'enable','off');
                        set(Resol,'backgroundcolor',XX.Clr.LightGray,'enable','off');
                        Method=1;
                    case 0 % cannot be combined with painters, e.g. bitmap
                        set(Painter,'value',0,'enable','off');
                        set(ZBuf,'value',1,'enable','on');
                        set(Resol,'backgroundcolor',XX.Clr.White,'string',num2str(DPI),'enable','on');
                        Method=2;
                    otherwise
                        set(Painter,'enable','on');
                        set(ZBuf,'enable','on');
                end
                switch PL{PrtID,4}
                    case 0 % NEVER
                        set(Color,'enable','off','value',0);
                        Clr=0;
                    case 1 % USER SPECIFIED (DEFAULT ON)
                        set(Color,'enable','on','value',1);
                        Clr=1;
                    case 2 % ALLWAYS
                        set(Color,'enable','off','value',1);
                        Clr=1;
                end
            case 'Options'
                % call windows printer setup dialog ...
                print -dsetup
                % bring md_print dialog back to front ...
                figure(fig)
            case 'Zbuffer'
                set(Painter,'value',0);
                set(ZBuf,'value',1);
                set(Resol,'backgroundcolor',XX.Clr.White,'string',num2str(DPI),'enable','on');
                Method=2;
            case 'DPI'
                X=eval(get(Resol,'string'),'NaN');
                if isnumeric(X) && isequal(size(X),[1 1]) && (round(X)==X)
                    if X<50
                        DPI=50;
                    elseif X>2400
                        DPI=2400;
                    else
                        DPI=X;
                    end
                end
                set(Resol,'string',num2str(DPI));
            case 'Painters'
                set(Painter,'value',1);
                set(ZBuf,'value',0);
                set(Resol,'backgroundcolor',XX.Clr.LightGray,'enable','off');
                Method=1;

            case 'Color'
                Clr=get(Color,'value');
            case 'InvertHardcopy'
                InvertHardcopy=get(InvHard,'value');
            case 'ApplyAll'
                ApplyAll=get(AllFig,'value');
        end
    end
end
if ishandle(fig)
    delete(fig);
end

Settings.PrtID=PrtID;
Settings.Method=Method;
Settings.DPI=DPI;
Settings.AllFigures=ApplyAll;
Settings.Color=Clr;
Settings.InvertHardcopy=InvertHardcopy;

if Cancel
    Settings.PrtID=0;
end

function add_bookmark(fname, bookmark_text, append)
% Adds a bookmark to the temporary EPS file after %%EndPageSetup
% Derived from BSD licensed export_fig by Oliver Woodford from MATLAB Central
% Based on original idea by Petr Nechaev

% Read in the file
fh = fopen(fname, 'r');
if fh == -1
    error('File %s not found.', fname);
end
try
    fstrm = fread(fh, '*char')';
catch ex
    fclose(fh);
    rethrow(ex);
end
fclose(fh);

if nargin<3 || ~append
    % Include standard pdfmark prolog to maximize compatibility
    fstrm = strrep(fstrm, '%%BeginProlog', sprintf('%%%%BeginProlog\n/pdfmark where {pop} {userdict /pdfmark /cleartomark load put} ifelse'));
end
pages = findstr(fstrm, '%%EndPageSetup');
lastpage = pages(end);
% Add page bookmark
fstrm = [fstrm(1:lastpage-1) strrep(fstrm(lastpage:end), '%%EndPageSetup', sprintf('%%%%EndPageSetup\n[ /Title (%s) /OUT pdfmark',bookmark_text))];

% Write out the updated file
fh = fopen(fname, 'w');
if fh == -1
    error('Unable to open %s for writing.', fname);
end
try
    fwrite(fh, fstrm, 'char*1');
catch ex
    fclose(fh);
    rethrow(ex);
end
fclose(fh);
