function [gX,normgX] = calcgx_df(net,X,PD,BZ,IWZ,LWZ,N,Ac,El,perf,Q,TS,dF,ro);
%CALCGX Calculate weight and bias performance gradient as a single vector.
%
%  Syntax
%
%    [gX,normgX] = calcgx(net,X,Pd,BZ,IWZ,LWZ,N,Ac,El,perf,Q,TS);
%
%  Description
%
%    This function calculates the gradient of a network's performance
%    with respect to its vector of weight and bias values X.
%
%    If the network has no layer delays with taps greater than 0
%    the result is the true gradient.
%
%    If the network as layer delays greater than 0, the result is
%    the Elman gradient, an approximation of the true gradient.
%
%    [gX,normgX] = CALCGX(NET,X,Pd,BZ,IWZ,LWZ,N,Ac,El,perf,Q,TS) takes,
%      NET    - Neural network.
%      X      - Vector of weight and bias values.
%      Pd     - Delayed inputs.
%      BZ     - Concurrent biases.
%      IWZ    - Weighted inputs.
%      LWZ    - Weighted layer outputs.
%      N      - Net inputs.
%      Ac     - Combined layer outputs.
%      El     - Layer errors.
%      perf   - Network performance.
%      Q      - Concurrent size.
%      TS     - Time steps.
%    and returns,
%      gX     - Gradient dPerf/dX.
%      normgX - Norm of gradient.
%
%======================================================================


dPerformFcn = feval(net.performFcn,'deriv');
gE = feval(dPerformFcn,'e',El,X,perf,net.performParam);
[gB,gIW,gLW] = calcgrad(net,Q,PD,BZ,IWZ,LWZ,N,Ac,gE,TS);

%Add Gradient of Derivative Error
[gBdf,gIWdf,gLWdf]=calcgrad_df(net,PD,Ac,dF);


c1 = 1;
%c2 = 1e4*(1/dF.SqMaxRatio);    %Old version ro.
c2 = ro;
for k=1:length(gB),
    gB{k} = c1*gB{k} - c2*gBdf{k};
end
gIW{1,1} = c1*gIW{1,1} - c2*gIWdf{1,1};
for k=1:net.numLayers-1,
    gLW{k+1,k} = c1*gLW{k+1,k} - c2*gLWdf{k+1,k};
end
%=========================================================================

gX = formgx(net,gB,gIW,gLW) + feval(dPerformFcn,'x',El,X,perf,net.performParam);
normgX = sqrt(sum(sum(gX.^2)));
