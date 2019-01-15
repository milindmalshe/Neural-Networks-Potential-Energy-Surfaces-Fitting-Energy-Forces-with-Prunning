

dat=xlsread('vBr_468Points_MP4eV_GreaterThan_18eV.xlsx');

dat= dat';
size(dat)
dat(:,1);
format long g

total= size(dat,2);

frac_TR= 1.0;
frac_VD= 1-frac_TR;

num_TR= floor(frac_TR*total);
num_VD= total-num_TR;

% mmm.TR{1}
dat_TR= dat(:,1:num_TR);
dat_VD= dat(:,(num_TR+1):(num_TR+num_VD));


size(dat_TR)
in_TR= dat_TR(1:12,:);
out_TR= dat_TR(26,:);

d_out_TR= dat_TR(27:38,:);

d_out_TR= -1.*d_out_TR;  % Force is a -ve of derivative, so multiply by -1


minmax(in_TR)
% mmm.TR{1}
% P=in_TR;
% T=out_TR;
[P_TR,minp,maxp,T_TR,mint,maxt] = premnmx(in_TR,out_TR);
minp;

% pwd
% cd ..

size(d_out_TR)
Df_TR= d_out_TR;


% Df_vect=reshape(Df,1,12*11947);
Df_vect_TR= reshape(Df_TR,1,(size(Df_TR,1)*size(Df_TR,2)) );

size(Df_vect_TR)

DfN_TR = tramnmxDf(Df_vect_TR,minp,maxp,mint,maxt);
% DfN = tramnmxDf(Df,minp,maxp,mint,maxt);
% DfN2 = tramnmxDf(Df,minp,maxp,mint,maxt);



% tempm=((kron((maxp-minp)',oneSMQ)./((maxt-mint)*ones(1,QR))));
% size(kron((maxp-minp)',oneSMQ))
size(DfN_TR)

DfN_TR = tramnmxDf(Df_vect_TR,minp,maxp,mint,maxt);


size(DfN_TR)
% mmm.TR{1}
% save 'vBr_dataM' P T Df_vect minp maxp mint maxt;
pwd
DfMap_TR= ones(1,(size(Df_TR,1)*size(Df_TR,2)) );

% save 'vBr_dataM' P T Df_vect minp maxp mint maxt DfMap;

% TR{1}.P=P_TR; TR{1}.T=T_TR; TR{1}.Df=Df_vect_TR; TR{1}.minp=minp; TR{1}.maxp=maxp; TR{1}.mint=mint;TR{1}.maxt=maxt; TR{1}.DfMap=DfMap_TR;
TR{1}.P=P_TR; TR{1}.T=T_TR; TR{1}.Df= DfN_TR; TR{1}.minp=minp; TR{1}.maxp=maxp; TR{1}.mint=mint;TR{1}.maxt=maxt; TR{1}.DfMap=DfMap_TR;





% % % % % % % 

size(dat_VD)
in_VD= dat_VD(1:12,:);
out_VD= dat_VD(26,:);

d_out_VD= dat_VD(27:38,:);
minmax(in_VD);
% mmm.TR{1}
% P=in_TR;
% T=out_TR;
% [P,minp,maxp,T,mint,maxt] = premnmx(in_VD,out_VD);

[P_VD]= tramnmx(in_VD,minp,maxp);
[T_VD]= tramnmx(out_VD,mint,maxt);


minp;

% pwd
% cd ..

size(d_out_VD)
Df_VD= d_out_VD;


% Df_vect=reshape(Df,1,12*11947);
Df_vect_VD= reshape(Df_VD,1,(size(Df_VD,1)*size(Df_VD,2)) );

size(Df_vect_VD)

DfN_VD = tramnmxDf(Df_vect_VD,minp,maxp,mint,maxt);
% DfN = tramnmxDf(Df,minp,maxp,mint,maxt);
% DfN2 = tramnmxDf(Df,minp,maxp,mint,maxt);



% tempm=((kron((maxp-minp)',oneSMQ)./((maxt-mint)*ones(1,QR))));
% size(kron((maxp-minp)',oneSMQ))
size(DfN_VD)

DfN_VD = tramnmxDf(Df_vect_VD,minp,maxp,mint,maxt);


size(DfN_VD)
% mmm.TR{1}
% save 'vBr_dataM' P T Df_vect minp maxp mint maxt;
pwd
DfMap_VD= ones(1,(size(Df_VD,1)*size(Df_VD,2)) );

% save 'vBr_dataM' P T Df_vect minp maxp mint maxt DfMap;

% VD.P=P_VD; VD.T=T_VD; VD.Df=Df_vect_VD; VD.DfMap=DfMap_VD; %VD.minp=minp; VD.maxp=maxp; VD.mint=mint; VD.maxt=maxt; 
VD{1}.P= []; VD{1}.T= []; VD{1}.Df= []; VD{1}.DfMap= [];

save 'vBr_468Points_MP4eV_GreaterThan_18eV_TR_VD' TR VD;

% save 'TRM' TR;




