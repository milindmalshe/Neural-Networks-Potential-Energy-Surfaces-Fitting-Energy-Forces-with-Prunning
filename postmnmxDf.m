function Df = postmnmxDf(DfN,minp,maxp,mint,maxt)

%Description
%THIS FUNCTION IS TO POSTPROCESS THE DERIVATIVE INFORMATION, WHICH WAS
%PREPROCESSED BY 'tramnmxDf'. THIS SCALES THE NORMALIZED DERIVATIVE
%BACK TO THE ORIGINAL SCALE.

%Parameters
%See 'tramnmxDf'.

%------------------------------------------------------------

[SM,QR]=size(DfN);
R = length(minp);
Q = QR/R;
oneSMQ = ones(SM,Q);

Df = (((maxt-mint)*ones(1,QR))./(kron((maxp-minp)',oneSMQ))).*DfN;