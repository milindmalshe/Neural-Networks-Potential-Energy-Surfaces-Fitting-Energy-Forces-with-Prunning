function [net,tr,Ac,El,v5,v6,v7,v8] = ...
  trainlm_df(net,Pd,Tl,Ai,Q,TS,VV,TV,v9,v10,v11,v12)
%TRAINLM Levenberg-Marquardt backpropagation.
%=====================================

% **[ NNT2 Support ]**
if ~isa(net,'struct') & ~isa(net,'char')
  nntobsu('trainlm','Use NNT2FF and TRAIN to update and train your network.')
  switch(nargin)
  case 5, [net,tr,Ac,El] = tlm1(net,Pd,Tl,Ai,Q); return
  case 6, [net,tr,Ac,El] = tlm1(net,Pd,Tl,Ai,Q,TS); return
  case 8, [net,tr,Ac,El,v5,v6] = tlm2(net,Pd,Tl,Ai,Q,TS,VV,TV); return
  case 9, [net,tr,Ac,El,v5,v6] = tlm2(net,Pd,Tl,Ai,Q,TS,VV,TV,v9); return
  case 11, [net,tr,Ac,El,v5,v6,v7,v8] = tlm3(net,Pd,Tl,Ai,Q,TS,VV,TV,v9,v10,v11); return
  case 12, [net,tr,Ac,El,v5,v6,v7,v8] = tlm3(net,Pd,Tl,Ai,Q,TS,VV,TV,v9,v10,v11,v12); return
  end
end

% FUNCTION INFO
% =============

if isstr(net)
  switch (net)
    case 'pnames',
    net = fieldnames(trainlm('pdefaults'));
    case 'pdefaults',
    trainParam.epochs = 100;
    trainParam.goal = 0;
    trainParam.max_fail = 5;
    trainParam.mem_reduc = 1;
    trainParam.min_grad = 1e-10;
    trainParam.mu = 0.001;
    trainParam.mu_dec = 0.1;
    trainParam.mu_inc = 10;
    trainParam.mu_max = 1e10;
    trainParam.show = 25;
    trainParam.time = inf;


    trainParam.MR_R = 0;
%--------------------------------------------------------

    trainParam.slopechk = 1e6;
%--------------------------------------------------------

    net = trainParam;
    otherwise,
    error('Unrecognized code.')
  end
  return
end

% CALCULATION
% ===========


dFtr.Value = VV.Dftr;
dFtr.Flag = VV.DftrMap;
dFtr.SqMaxRatio = VV.SqMaxRatio;
if VV.DfvdFlag,
    dFvd.Value = VV.Dfvd;
    dFvd.Flag = VV.DfvdFlag;
    dFvd.SqMaxRatio = dFtr.SqMaxRatio;
else
    VV = [];
end
if net.trainParam.mem_reduc > 1,
   MR_R = 1;    %Activate memory/speed tradeoff in derivative calc.
else
   MR_R = net.trainParam.MR_R; %Default value = 0.
end                      
ro = 1e4*(1/dFtr.SqMaxRatio);
%--------------------------------------------------------

% Parameters
epochs = net.trainParam.epochs;
goal = net.trainParam.goal;
max_fail = net.trainParam.max_fail;
mem_reduc = net.trainParam.mem_reduc;
min_grad = net.trainParam.min_grad;
mu = net.trainParam.mu;
mu_inc = net.trainParam.mu_inc;
mu_dec = net.trainParam.mu_dec;
mu_max = net.trainParam.mu_max;
show = net.trainParam.show;
time = net.trainParam.time;

slope_stop = net.trainParam.slopechk;
%--------------------------------------------------------------------- 

% Parameter Checking
if (~isa(epochs,'double')) | (~isreal(epochs)) | (any(size(epochs)) ~= 1) | ...
  (epochs < 1) | (round(epochs) ~= epochs)
  error('Epochs is not a positive integer.')
end
if (~isa(goal,'double')) | (~isreal(goal)) | (any(size(goal)) ~= 1) | ...
  (goal < 0)
  error('Goal is not zero or a positive real value.')
end
if (~isa(max_fail,'double')) | (~isreal(max_fail)) | (any(size(max_fail)) ~= 1) | ...
  (max_fail < 1) | (round(max_fail) ~= max_fail)
  error('Max_fail is not a positive integer.')
end
if (~isa(mem_reduc,'double')) | (~isreal(mem_reduc)) | (any(size(mem_reduc)) ~= 1) | ...
  (mem_reduc < 1) | (round(mem_reduc) ~= mem_reduc)
  error('Mem_reduc is not a positive integer.')
end
if (~isa(min_grad,'double')) | (~isreal(min_grad)) | (any(size(min_grad)) ~= 1) | ...
  (min_grad < 0)
  error('Min_grad is not zero or a positive real value.')
end
if (~isa(mu,'double')) | (~isreal(mu)) | (any(size(mu)) ~= 1) | ...
  (mu <= 0)
  error('Mu is not a positive real value.')
end
if (~isa(mu_dec,'double')) | (~isreal(mu_dec)) | (any(size(mu_dec)) ~= 1) | ...
  (mu_dec < 0) | (mu_dec > 1)
  error('Mu_dec is not a real value between 0 and 1.')
end
if (~isa(mu_inc,'double')) | (~isreal(mu_inc)) | (any(size(mu_inc)) ~= 1) | ...
  (mu_inc < 1)
  error('Mu_inc is not a real value greater than 1.')
end
if (~isa(mu_max,'double')) | (~isreal(mu_max)) | (any(size(mu_max)) ~= 1) | ...
  (mu_max <= 0)
  error('Mu_max is not a positive real value.')
end
if (mu > mu_max)
  error('Mu is greater than Mu_max.')
end
if (~isa(show,'double')) | (~isreal(show)) | (any(size(show)) ~= 1) | ...
  (isfinite(show) & ((show < 1) | (round(show) ~= show)))
  error('Show is not ''NaN'' or a positive integer.')
end
if (~isa(time,'double')) | (~isreal(time)) | (any(size(time)) ~= 1) | ...
  (time < 0)
  error('Time is not zero or a positive real value.')
end

% Constants
this = 'TRAINLM_df';
doValidation = ~isempty(VV);
doTest = ~isempty(TV);

% Initialize
flag_stop=0;
stop = '';
startTime = clock;
X = getx(net);
numParameters = length(X);
ii = sparse(1:numParameters,1:numParameters,ones(1,numParameters));


%[perf,El,Ac,N,Zb,Zi,Zl] = calcperf(net,X,Pd,Tl,Ai,Q,TS);
%ro = 1;
[perf,El,Ac,N,Zb,Zi,Zl] = calcperf_df(net,X,Pd,Tl,Ai,Q,TS,dFtr,ro);
%--------------------------------------------------------

if (doValidation)
  VV.net = net;


  %vperf = calcperf(net,X,VV.Pd,VV.Tl,VV.Ai,VV.Q,VV.TS);
  
  vperf = calcperf_df(net,X,VV.Pd,VV.Tl,VV.Ai,VV.Q,VV.TS,dFvd,ro);
%--------------------------------------------------------
  
  VV.perf = vperf;
  VV.numFail = 0;
end


%tr = newtr(epochs,'perf','vperf','tperf','mu');
tr = newtr(epochs,'perf','vperf','vperf_f','vperf_df','tperf','perf_f','perf_df','normgX','normgX_df','ro','etime');
%--------------------------------------------------------

% Train
for epoch=0:epochs
    

%    if perf.F <= (perf.Df*ED),
%        ro = 10*ro;
%        if ro > 1e10;
%            ro = 1e10;
%        end
%    else
%        ro = ro/10;
%        if ro < 1e-10,
%            ro = 1e-10;
%        end
%    end 
%---------------------------------------------------------

  % Jacobian
  [je,jj,normgX]=calcjejj(net,Pd,Zb,Zi,Zl,N,Ac,El,Q,TS,mem_reduc);
  

  % Jacobian Derivative
  [je_df,jj_df,normgX_df]=calcjejj_df(net,Pd,Ac,dFtr,mem_reduc,MR_R);

  %Total Jacobian
  c1 = 1;
  %c2 = 1e4*(1/dFtr.SqMaxRatio);
  c2 = ro;
  je = c1*je + c2*je_df;
  jj = c1*jj + c2*jj_df;
  normgX = sqrt(je'*je); 
%--------------------------------------------------------

  % Training Record
  epochPlus1 = epoch+1;


%  tr.perf(epoch+1) = perf;
%  tr.mu(epoch+1) = mu;
%  if (doValidation)
%    tr.vperf(epochPlus1) = VV.perf;
%  end

  tr.perf(epoch+1) = perf.T;
  tr.perf_f(epoch+1) = perf.F;
  tr.perf_df(epoch+1) = perf.Df;
  tr.normgX(epoch+1) = normgX;
  tr.normgX_df(epoch+1) = normgX_df;
  tr.mu(epoch+1) = mu;
  tr.ro(epoch+1) = ro;
  if (doValidation)
    tr.vperf(epochPlus1) = VV.perf.T;
    tr.vperf_f(epochPlus1) = VV.perf.F;
    tr.vperf_df(epochPlus1) = VV.perf.Df;
  end
%---------------------------------------------------------------------

  if (doTest)
    tr.tperf(epochPlus1) = calcperf(net,X,TV.Pd,TV.Tl,TV.Ai,TV.Q,TV.TS);
  end
  
  % Stopping Criteria
  currentTime = etime(clock,startTime);
  

  tr.etime(epoch+1) = currentTime;
%--------------------------------------------------------  
  


  %if (perf <= goal)
  if (perf.T <= goal)
%--------------------------------------------------------

    stop = 'Performance goal met.';
  elseif (epoch == epochs)
    stop = 'Maximum epoch reached, performance goal was not met.';
  elseif (currentTime > time)
    stop = 'Maximum time elapsed, performance goal was not met.';
  elseif (normgX < min_grad)
    stop = 'Minimum gradient reached, performance goal was not met.';
  elseif (mu > mu_max)
    stop = 'Maximum MU reached, performance goal was not met.';
  elseif (doValidation) & (VV.numFail > max_fail)
    stop = 'Validation stop.';
  elseif flag_stop
    stop = 'User stop.';

  elseif isequal(epoch,slope_stop),
    stop = 'Slope check';
%------------------------------------------------------------- 
  end
  
  % Progress
  if isfinite(show) & (~rem(epoch,show) | length(stop))
    fprintf(this);
  if isfinite(epochs) fprintf(', Epoch %g/%g',epoch, epochs); end
  if isfinite(time) fprintf(', Time %4.1f%%',currentTime/time*100); end
  

  %if isfinite(goal) fprintf(', %s %g/%g',upper(net.performFcn),perf,goal); end
  %if isfinite(goal) fprintf(', %s %g/%g',upper(net.performFcn),perf.T,goal); end
  if isfinite(goal) fprintf(', %s %g/%g',upper(net.performFcn),perf.F,goal); end
  if isfinite(goal) fprintf(', %s %g/%g',upper(net.performFcn),perf.Df,goal); end %May 24, 08
%---------------------------------------------------------------------

  if isfinite(min_grad) fprintf(', Gradient %g/%g',normgX,min_grad); end
  fprintf('\n')
  flag_stop=plotperf(tr,goal,this,epoch);
    if length(stop) fprintf('%s, %s\n\n',this,stop); end
  end
 
  % Stop when criteria indicate its time
  if length(stop)
    if (doValidation)
    net = VV.net;
  end
    break
  end
  
  % Levenberg Marquardt
  while (mu <= mu_max)
    dX = -(jj+ii*mu) \ je;
    X2 = X + dX;
    net2 = setx(net,X2);


%    [perf2,El2,Ac2,N2,Zb2,Zi2,Zl2] = calcperf(net2,X2,Pd,Tl,Ai,Q,TS);
%    if (perf2 < perf)
    
    [perf2,El2,Ac2,N2,Zb2,Zi2,Zl2] = calcperf_df(net2,X2,Pd,Tl,Ai,Q,TS,dFtr,ro);
    if (perf2.T < perf.T)
%--------------------------------------------------------

      X = X2; net = net2; Zb = Zb2; Zi = Zi2; Zl = Zl2;
      N = N2; Ac = Ac2; El = El2; perf = perf2;
      mu = mu * mu_dec;
      if (mu < 1e-20)
        mu = 1e-20;
      end
      break
    end
    mu = mu * mu_inc;
  end

  % Validation
  if (doValidation)


%    vperf = calcperf(net,X,VV.Pd,VV.Tl,VV.Ai,VV.Q,VV.TS);
%  if (vperf < VV.perf)
%    VV.perf = vperf; VV.net = net; VV.numFail = 0;
%  elseif (vperf > VV.perf)

    vperf = calcperf_df(net,X,VV.Pd,VV.Tl,VV.Ai,VV.Q,VV.TS,dFvd,ro);
  if (vperf.T < VV.perf.T)
    VV.perf = vperf; VV.net = net; VV.numFail = 0;
  elseif (vperf.T > VV.perf.T)
%--------------------------------------------------------

      VV.numFail = VV.numFail + 1;
  end
  end
end

% Finish
tr = cliptr(tr,epoch);
