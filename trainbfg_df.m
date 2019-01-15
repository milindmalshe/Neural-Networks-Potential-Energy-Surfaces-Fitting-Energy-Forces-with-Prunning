function [net,tr,Ac,El] = trainbfg_df(net,Pd,Tl,Ai,Q,TS,VV,TV)
%TRAINBFG BFGS quasi-Newton backpropagation.
%======================================================================


if isstr(net)
  switch (net)
    case 'pnames',
    net = {'epochs','show','goal','time','min_grad','max_fail','searchFcn','scal_tol','alpha',...
           'beta','delta','gama','low_lim','up_lim','maxstep','minstep','bmax'};
    case 'pdefaults',
    trainParam.epochs = 100;
    trainParam.show = 25;
    trainParam.goal = 0;
    trainParam.time = inf;
    trainParam.min_grad = 1.0e-6;
    trainParam.max_fail = 5;

    %trainParam.searchFcn = 'srchbac';
    trainParam.searchFcn = 'srchbac_df';
%======================================================================    
    trainParam.scale_tol = 20;
    trainParam.alpha = 0.001;
    trainParam.beta = 0.1;
    trainParam.delta = 0.01;
    trainParam.gama = 0.1;
    trainParam.low_lim = 0.1;
    trainParam.up_lim = 0.5;
    trainParam.maxstep = 100;
    trainParam.minstep = 1.0e-6;
    trainParam.bmax = 26;

    trainParam.retrain = 0;
    trainParam.initdX = 0;
    trainParam.initdperf = 0;
    trainParam.H = 0;
%======================================================================   

    trainParam.slopechk = 1e6;
%====================================================================== 
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
ro = 1e4*(1/dFtr.SqMaxRatio);
%======================================================================================


% Constants
this = 'TRAINBFG';
epochs = net.trainParam.epochs;
show = net.trainParam.show;
goal = net.trainParam.goal;
time = net.trainParam.time;
min_grad = net.trainParam.min_grad;
max_fail = net.trainParam.max_fail;
scale_tol = net.trainParam.scale_tol;
delta = net.trainParam.delta;
searchFcn = net.trainParam.searchFcn;
tol = delta/scale_tol;
doValidation = ~isempty(VV);
doTest = ~isempty(TV);
retcode = 0;

retrain = net.trainParam.retrain;
init_dX = net.trainParam.initdX;
init_dperf = net.trainParam.initdperf;
H_temp = net.trainParam.H;
%======================================================================

slope_stop = net.trainParam.slopechk;
%======================================================================

% Parameter Checking
if (~isa(epochs,'double')) | (~isreal(epochs)) | (any(size(epochs)) ~= 1) | ...
  (epochs < 1) | (round(epochs) ~= epochs)
  error('Epochs is not a positive integer.')
end
if (~isa(show,'double')) | (~isreal(show)) | (any(size(show)) ~= 1) | ...
  (isfinite(show) & ((show < 1) | (round(show) ~= show)))
  error('Show is not ''NaN'' or a positive integer.')
end
if (~isa(goal,'double')) | (~isreal(goal)) | (any(size(goal)) ~= 1) | ...
  (goal < 0)
  error('Goal is not zero or a positive real value.')
end
if (~isa(time,'double')) | (~isreal(time)) | (any(size(time)) ~= 1) | ...
  (time < 0)
  error('Time is not zero or a positive real value.')
end
if (~isa(min_grad,'double')) | (~isreal(min_grad)) | (any(size(min_grad)) ~= 1) | ...
  (min_grad < 0)
  error('Min_grad is not zero or a positive real value.')
end
if (~isa(max_fail,'double')) | (~isreal(max_fail)) | (any(size(max_fail)) ~= 1) | ...
  (max_fail < 1) | (round(max_fail) ~= max_fail)
  error('Max_fail is not a positive integer.')
end
if(isstr(searchFcn))
  exist_search = exist(searchFcn);
  if (exist_search<2) | (exist_search>3)
    error('SearchFcn is not a valid search function.')
  end
else
  error('SearchFcn is not a character string')
end
if (~isa(scale_tol,'double')) | (~isreal(scale_tol)) | (any(size(scale_tol)) ~= 1) | ...
  (scale_tol <= 0)
  error('Scale_tol is not a positive real value.')
end
if (~isa(delta,'double')) | (~isreal(delta)) | (any(size(delta)) ~= 1) | ...
  (delta <= 0)
  error('Delta is not a positive real value.')
end

% Initialize
flag_stop = 0;
stop = '';
startTime = clock;
X = getx(net);
num_X = length(X);
if (doValidation)
  VV.net = net;
  VV.X = X;

  %VV.perf = calcperf(net,X,VV.Pd,VV.Tl,VV.Ai,VV.Q,VV.TS);
  VV.perf = calcperf_df(net,X,VV.Pd,VV.Tl,VV.Ai,VV.Q,VV.TS,dFvd,ro);
%=============================================================   
  vperf = VV.perf;
  VV.numFail = 0;
end
tr.epoch = 0:epochs;

%tr = newtr(epochs,'perf','vperf','tperf');
%tr = newtr(epochs,'perf','vperf','tperf','perf_f','perf_df','normgX');  %Aug 18, 06
%tr = newtr(epochs,'perf','vperf','vperf_df','tperf','perf_f','perf_df','normgX'); %Aug 23, 06
tr = newtr(epochs,'perf','vperf','vperf_f','vperf_df','tperf','perf_f','perf_df','normgX','ro','etime'); %July 12, 08
%=============================================================
flops(0)

a=0;

for epoch=0:epochs

  epochPlus1 = epoch+1;

  % Performance, Gradient and Search Direction

  if (epoch == 0)

    % First iteration
    
%==================ADDED Mar 21, 2008=========================
    %ro = 1;
%=============================================================

    % Initial performance

    %[perf,El,Ac,N,Zb,Zi,Zl] = calcperf(net,X,Pd,Tl,Ai,Q,TS);
    [perf,El,Ac,N,Zb,Zi,Zl] = calcperf_df(net,X,Pd,Tl,Ai,Q,TS,dFtr,ro);
%=============================================================    
    perf_old = perf;

    %ch_perf = perf;
    ch_perf = perf.T;
%=============================================================     
    avg1 = 0; avg2 = 0; sum1 = 0; sum2 = 0;

    % Intial gradient and norm of gradient

    %gX = -calcgx(net,X,Pd,Zb,Zi,Zl,N,Ac,El,perf,Q,TS);
    gX = -calcgx_df(net,X,Pd,Zb,Zi,Zl,N,Ac,El,perf,Q,TS,dFtr,ro);
%=============================================================      
    normgX = sqrt(gX'*gX);
    gX_old = gX;

    % Initial search direction and initial slope
    II = eye(num_X);
    H = II;

    %dX  = -gX;
    %dperf = gX'*dX;
    if ~retrain,
        dX  = -gX;
        dperf = gX'*dX;
    else
        dX = init_dX;
        dperf = init_dperf;
        H = H_temp;
    end
%=====================================================================

  else

    % After first iteration

%====================ADDED Mar 21, 2008=======================
    %if epochPlus1 == 2,
    %    ro = 1;
    %else
    %    if tr.perf_f(epochPlus1-1) > tr.perf_f(epochPlus1-2),
    %        ro = ro/10;
    %    elseif tr.perf_df(epochPlus1-1) > tr.perf_df(epochPlus1-2),
    %        ro = ro*10;
    %    else
    %        ro = ro;
    %    end
    %end
%============================================================= 
    
    % Calculate change in gradient
    dgX = gX - gX_old;

    % Calculate change in performance and save old performance

    %ch_perf = perf - perf_old;
    ch_perf = perf.T - perf_old.T;
%=============================================================       
    perf_old = perf;
  
    % Calculate new Hessian approximation
    H = H + gX_old*gX_old'/(gX_old'*dX) + dgX*dgX'/(dgX'*X_step);

    % Calculate new search direction
    dX = -H\gX;

    % Check for a descent direction
    dperf = gX'*dX;
    if dperf>0
      H = II;
      dX = -gX;
      dperf = gX'*dX;
    end

    % Save old gradient and norm of gradient
    normgX = sqrt(gX'*gX);
    gX_old = gX;

  end

  % Training Record
  currentTime = etime(clock,startTime);

  %tr.perf(epochPlus1) = perf;
  tr.perf(epochPlus1) = perf.T;
  tr.perf_f(epochPlus1) = perf.F;
  tr.perf_df(epochPlus1) = perf.Df;
  tr.normgX(epochPlus1) = normgX;
  tr.ro(epochPlus1) = ro;
  tr.etime(epochPlus1) = currentTime;
%=============================================================================   
  if (doValidation)

    %tr.vperf(epochPlus1) = vperf;
    %tr.vperf(epochPlus1) = vperf.T;    %August 18, 2006
    %tr.vperf(epochPlus1) = vperf.F;
    %tr.vperf_df(epochPlus1) = vperf.Df; %August 23, 2006
    tr.vperf(epochPlus1) = vperf.T;
    tr.vperf_f(epochPlus1) = vperf.F;
    tr.vperf_df(epochPlus1) = vperf.Df; %September 13, 2006
%=============================================================    
  end
  if (doTest)
    tr.tperf(epochPlus1) = calcperf(net,X,TV.Pd,TV.Tl,TV.Ai,TV.Q,TV.TS);
  end
 
  % Stopping Criteria

  %if (perf <= goal)
  if (perf.T <= goal)
  %if (perf.F <= goal)
%=============================================================      
    stop = 'Performance goal met.';
  elseif (epoch == epochs)
    stop = 'Maximum epoch reached, performance goal was not met.';
  elseif (currentTime > time)
    stop = 'Maximum time elapsed, performance goal was not met.';
  elseif(any(isnan(dX)) | any(isinf(dX)))
    stop =  'Precision problems in matrix inversion.';
  elseif (normgX < min_grad)
    stop = 'Minimum gradient reached, performance goal was not met.';
  elseif (doValidation) & (VV.numFail > max_fail)
    stop = 'Validation stop.';
  elseif flag_stop
    stop = 'User stop.';

  elseif isequal(epoch,slope_stop),
    stop = 'Slope check';
%=============================================================
  end
 
  % Progress
  if isfinite(show) & (~rem(epoch,show) | length(stop))
    fprintf('%s%s%s',this,'-',searchFcn);
  if isfinite(epochs) fprintf(', Epoch %g/%g',epoch, epochs); end
  if isfinite(time) fprintf(', Time %g%%',currentTime/time/100); end

  %if isfinite(goal) fprintf(', %s %g/%g',upper(net.performFcn),perf,goal); end
  %if isfinite(goal) fprintf(', %s %g/%g',upper(net.performFcn),perf.T,goal); end
  if isfinite(goal) fprintf(', %s %g/%g',upper(net.performFcn),perf.F,goal); end
  if isfinite(goal) fprintf(', %s %g/%g',upper(net.performFcn),perf.Df,goal); end
%============================================================================   
  if isfinite(min_grad) fprintf(', Gradient %g/%g',normgX,min_grad); end
  fprintf('\n')
    flag_stop = plotperf(tr,goal,this,epoch);
    if length(stop) fprintf('%s, %s\n\n',this,stop); end
  end
  if length(stop), break; end

  % Minimize the performance along the search direction
  delta = 1;

  %[a,gX,perf,retcode,delta,tol] =
  %feval(searchFcn,net,X,Pd,Tl,Ai,Q,TS,dX,gX,perf,dperf,delta,tol,ch_perf);
  [a,gX,perf,retcode,delta,tol] = feval(searchFcn,net,X,Pd,Tl,Ai,Q,TS,dX,gX,perf,dperf,delta,tol,ch_perf,dFtr,ro);
%============================================================= 
  % Keep track of the number of function evaluations
  sum1 = sum1 + retcode(1);
  sum2 = sum2 + retcode(2);
  avg1 = sum1/epochPlus1;
  avg2 = sum2/epochPlus1;

  % Update X
  X_step = a*dX;
  X = X + X_step;
  net = setx(net,X);
 
  % Validation
  if (doValidation)

  %  vperf = calcperf(net,X,VV.Pd,VV.Tl,VV.Ai,VV.Q,VV.TS);
  %if (vperf < VV.perf)
  %  VV.perf = vperf; VV.net = net; VV.X = X; VV.numFail = 0;
  %elseif (vperf > VV.perf)
     vperf = calcperf_df(net,X,VV.Pd,VV.Tl,VV.Ai,VV.Q,VV.TS,dFvd);
  if (vperf.T < VV.perf.T)
    VV.perf.T = vperf.T; VV.net = net; VV.X = X; VV.numFail = 0;
  elseif (vperf.T > VV.perf.T)
%=============================================================      
      VV.numFail = VV.numFail + 1;
  end
  end
  
end

if (doValidation)
  net = VV.net;
end


% Finish
tr = cliptr(tr,epoch);

%[perf,El,Ac] = calcperf(net,X,Pd,Tl,Ai,Q,TS);
[perf,El,Ac] = calcperf_df(net,X,Pd,Tl,Ai,Q,TS,dFtr,ro);
%=============================================================  


net.trainParam.initdX = dX;
net.trainParam.initdperf = dperf;
net.trainParam.H = H;
%=============================================================
