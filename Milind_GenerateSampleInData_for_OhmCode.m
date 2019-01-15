


in=0:0.01:2*pi;
out=sin(in);

plot(in,out,'.');

[P,minp,maxp,T,mint,maxt]=premnmx(in,out);

Df=cos(in);
hold on; plot(in,Df);

Df_vect= reshape(Df,1,(size(Df,1)*size(Df,2)) );
DfN = tramnmxDf(Df_vect,minp,maxp,mint,maxt);

DfMap= ones(1,(size(Df,1)*size(Df,2)) );


TR{1}.P=P; TR{1}.T=T; TR{1}.Df=Df_vect; TR{1}.minp=minp; TR{1}.maxp=maxp; TR{1}.mint=mint;TR{1}.maxt=maxt; TR{1}.DfMap=DfMap;

TR

TR{1}

VD{1}.P= []; VD{1}.T= []; VD{1}.Df= []; VD{1}.DfMap= [];


save 'TEST_sinX' TR VD;