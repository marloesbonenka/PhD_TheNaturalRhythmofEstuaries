function [XZ,YZ,T,MORFAC,TMAX,TMAX_tot] = loadtrim(trim)
% loadtrim: loads variables from trim file that are at least needed for
% plotting.
% 
% loadtrim - trimfile is inputfile

for i=1:size(trim,2);   
    %NMAX(i) = vs_let(trim(i),'map-const','NMAX','quiet');
    %MMAX(i) = vs_let(trim(i),'map-const','MMAX','quiet');
    if strcmp(trim(i).Format,'l');
        DT = vs_let(trim(i),'map-const','DT','quiet');
        TUNIT = vs_let(trim(i),'map-const','TUNIT','quiet');
        ITMAPC = vs_let(trim(i),'map-info-series','ITMAPC','quiet');
        MORFAC(i) = vs_let(trim(i),'map-infsed-serie',{1},'MORFAC','quiet');
        T{i} = ITMAPC*DT*TUNIT; 
        TMAX(i)=size(T{i},1);
        clear ITMAPC
    else
        DT = 0;
        TUNIT = 0;
        MORFAC(i) = 0;
        T{i} = 0;
        TMAX(i) = 0;
    end
    
end
    TMAX_tot = min(TMAX);
    XZ = (squeeze(vs_let(trim(1),'map-const','XZ','quiet')));
    YZ = (squeeze(vs_let(trim(1),'map-const','YZ','quiet')));
       
end
