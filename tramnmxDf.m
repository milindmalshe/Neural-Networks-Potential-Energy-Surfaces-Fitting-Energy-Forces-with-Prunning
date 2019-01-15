function DfN = tramnmxDf(Df,minp,maxp,mint,maxt);

%Description
%THIS FUNCTION IS TO NORMALIZE THE DERIVATIVE INFORMATION SO THAT
%IT WILL CONFORM TO THE SCALED INPUTS AND TARGETS.

%PARAMETERS:
%   Df      - Raw Derivative Information, size(SM,QxR).
%   minp/maxp/mint/maxt - These are obtained from premnmx().
%   DfN     - Normalized Derivative, size(SM,QxR).
%   DfTemp  - Temp variable, size (R,Q).
%   DfNTemp - Temp variable, size (R,Q).

%-----------------------------------------------------------

[SM,QR] = size(Df);
R = length(minp);
Q = QR/R;
oneSMQ = ones(SM,Q);

DfN = ((kron((maxp-minp)',oneSMQ)./((maxt-mint)*ones(1,QR)))).*Df;


    