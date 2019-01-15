function [a,gX,perfb,retcode1,delta,tol] = srchbac_df(net,X,Pd,Tl,Ai,Q,TS,dX,gX,perfa,dperfa,delta,tol,ch_perf,dF,ro)
%SRCHBAC One-dimensional minimization using backtracking.
%======================================================================

% ALGORITHM PARAMETERS
scale_tol = net.trainParam.scale_tol;
alpha = net.trainParam.alpha;
beta = net.trainParam.beta;
low_lim = net.trainParam.low_lim;
up_lim = net.trainParam.up_lim;
maxstep = net.trainParam.maxstep;
minstep = net.trainParam.minstep;
norm_dX = norm(dX);
minlambda = minstep/norm_dX;
maxlambda = maxstep/norm_dX;
cnt1 = 0;
cnt2 = 0;
start = 1;

% Parameter Checking
if (~isa(scale_tol,'double')) | (~isreal(scale_tol)) | (any(size(scale_tol)) ~= 1) | ...
  (scale_tol <= 0)
  error('Scale_tol is not a positive real value.')
end
if (~isa(alpha,'double')) | (~isreal(alpha)) | (any(size(alpha)) ~= 1) | ...
  (alpha < 0) | (alpha > 1)
  error('Alpha is not a real value between 0 and 1.')
end
if (~isa(beta,'double')) | (~isreal(beta)) | (any(size(beta)) ~= 1) | ...
  (beta < 0) | (beta > 1)
  error('Beta is not a real value between 0 and 1.')
end
if (~isa(low_lim,'double')) | (~isreal(low_lim)) | (any(size(low_lim)) ~= 1) | ...
  (low_lim < 0) | (low_lim > 1)
  error('Low_lim is not a real value between 0 and 1.')
end
if (~isa(up_lim,'double')) | (~isreal(up_lim)) | (any(size(up_lim)) ~= 1) | ...
  (up_lim < 0) | (up_lim > 1)
  error('Up_lim is not a real value between 0 and 1.')
end
if (~isa(maxstep,'double')) | (~isreal(maxstep)) | (any(size(maxstep)) ~= 1) | ...
  (maxstep <= 0)
  error('Maxstep is not a positive real value.')
end
if (~isa(minstep,'double')) | (~isreal(minstep)) | (any(size(minstep)) ~= 1) | ...
  (minstep <= 0)
  error('Minstep is not a positive real value.')
end

% TAKE INITIAL STEP
lambda = 1;
X_temp = X + lambda*dX;
net_temp = setx(net,X_temp);
  
% CALCULATE PERFORMANCE AT NEW POINT
%[perfb,E,Ac,N,Zb,Zi,Zl] = calcperf(net_temp,X_temp,Pd,Tl,Ai,Q,TS);
[perfb,E,Ac,N,Zb,Zi,Zl] = calcperf_df(net_temp,X_temp,Pd,Tl,Ai,Q,TS,dF,ro);
%=============================================================    
g_flag = 0;
cnt1 = cnt1 + 1;

count = 0;
% MINIMIZE ALONG A LINE (BACKTRACKING)
retcode = 4;

while retcode>3
  
  count=count+1;


  %if (perfb <= perfa + alpha*lambda*dperfa)         %CONDITION ALPHA IS SATISFIED
  if (perfb.T <= perfa.T + alpha*lambda*dperfa)
%============================================================= 

    if (g_flag == 0)

      %gX_temp = -calcgx(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb,Q,TS);
      %gX_temp = -calcgx_df(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb.T,Q,TS,dF);
      gX_temp = -calcgx_df(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb,Q,TS,dF,ro);
%===================================================================       
      dperfb = gX_temp'*dX;
    end
  
    if (dperfb < beta * dperfa)                     %CONDITION BETA IS NOT SATISFIED

    if (start==1) & (norm_dX<maxstep)

    %while (perfb<=perfa+alpha*lambda*dperfa)&(dperfb<beta*dperfa)&(lambda<maxlambda)
    while (perfb.T<=perfa.T+alpha*lambda*dperfa)&(dperfb<beta*dperfa)&(lambda<maxlambda)
%============================================================= 
          % INCREASE STEP SIZE UNTIL BETA CONDITION IS SATISFIED

          lambda_old = lambda;
          perfb_old = perfb;
          lambda = min ([2*lambda maxlambda]);
          X_temp = X + lambda*dX;
          net_temp = setx(net,X_temp);

          %[perfb,E,Ac,N,Zb,Zi,Zl] = calcperf(net_temp,X_temp,Pd,Tl,Ai,Q,TS);
          [perfb,E,Ac,N,Zb,Zi,Zl] = calcperf_df(net_temp,X_temp,Pd,Tl,Ai,Q,TS,dF,ro);
%=============================================================           
          cnt1 = cnt1 + 1;
          g_flag = 0;

          %if (perfb <= perfa+alpha*lambda*dperfa)           
          %  gX_temp = -calcgx(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb,Q,TS);
          %if (perfb.T <= perfa.T+alpha*lambda*dperfa)           
            %gX_temp = -calcgx_df(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb.T,Q,TS,dF);
          if (perfb.T <= perfa.T+alpha*lambda*dperfa)
            gX_temp = -calcgx_df(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb,Q,TS,dF,ro);
%============================================================================             
            dperfb = gX_temp'*dX;
            g_flag = 1;
          end
    end
      end


      %if (lambda<1) | ((lambda>1)&(perfb>perfa+alpha*lambda*dperfa))
      if (lambda<1) | ((lambda>1)&(perfb.T>perfa.T+alpha*lambda*dperfa))
%=============================================================           
      lambda_lo = min([lambda lambda_old]);
    lambda_diff = abs(lambda_old - lambda);
    
        if (lambda < lambda_old)

    %      perf_lo = perfb;
    %      perf_hi = perfb_old;
    %else
    %      perf_lo = perfb_old;
    %      perf_hi = perfb;
          perf_lo = perfb.T;
          perf_hi = perfb_old.T;
    else
          perf_lo = perfb_old.T;
          perf_hi = perfb.T;
%============================================================= 
    end


        while (dperfb<beta*dperfa)&(lambda_diff>minlambda)
    
          lambda_incr=-dperfb*(lambda_diff^2)/(2*(perf_hi-(perf_lo+dperfb*lambda_diff)));
          if (lambda_incr<0.2*lambda_diff)
        lambda_incr=0.2*lambda_diff;
          end
      
          %UPDATE X
          lambda = lambda_lo + lambda_incr;
          X_temp = X + lambda*dX;
          net_temp = setx(net,X_temp);

          %[perfb,E,Ac,N,Zb,Zi,Zl] = calcperf(net_temp,X_temp,Pd,Tl,Ai,Q,TS);
          [perfb,E,Ac,N,Zb,Zi,Zl] = calcperf_df(net_temp,X_temp,Pd,Tl,Ai,Q,TS,dF,ro);
%=============================================================           
          g_flag = 0;
          cnt2 = cnt2 + 1;


      %if (perfb>perfa+alpha*lambda*dperfa)
      if (perfb.T>perfa.T+alpha*lambda*dperfa)
%=============================================================          
            lambda_diff = lambda_incr;

            %perf_hi = perfb;
            perf_hi = perfb.T;
%=============================================================             
      else

            %gX_temp = -calcgx(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb,Q,TS);
            %gX_temp = -calcgx_df(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb.T,Q,TS,dF);
            gX_temp = -calcgx_df(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb,Q,TS,dF,ro);
%============================================================================             
            dperfb = gX_temp'*dX;
            g_flag = 1;
            if (dperfb<beta*dperfa)
              lambda_lo = lambda;
              lambda_diff = lambda_diff - lambda_incr;

              %perf_lo = perfb;
              perf_lo = perfb.T;
%=============================================================               
            end
          end

        end
    
        retcode = 0;

        if (dperfb<beta*dperfa)    %COULDN'T SATISFY BETA CONDITION

          %perfb = perf_lo;
          perfb.T = perf_lo;
%=============================================================          
          lambda = lambda_lo;
          X_temp = X + lambda*dX;
          net_temp = setx(net,X_temp);

          %[perfb,E,Ac,N,Zb,Zi,Zl] = calcperf(net_temp,X_temp,Pd,Tl,Ai,Q,TS);
          [perfb,E,Ac,N,Zb,Zi,Zl] = calcperf_df(net_temp,X_temp,Pd,Tl,Ai,Q,TS,dF,ro);
%=============================================================           
          g_flag = 0;
          cnt2 = cnt2 + 1;
          retcode = 3;
        end
            
      end

      if (lambda*norm_dX>0.99*maxstep)    % MAXIMUM STEP TAKEN
    retcode = 2;
      end

    else
      
      retcode = 0;
    
      if (lambda*norm_dX>0.99*maxstep)    % MAXIMUM STEP TAKEN
        retcode = 2;
      end

    end

  elseif (lambda<minlambda)   % MINIMUM STEPSIZE REACHED

    retcode = 1;

  else    % CONDITION ALPHA IS NOT SATISFIED - REDUCE THE STEP SIZE

    if (start == 1)
      % FIRST BACKTRACK, QUADRATIC FIT

      %lambda_temp = -dperfa/(2*(perfb - perfa - dperfa));
      lambda_temp = -dperfa/(2*(perfb.T - perfa.T - dperfa));
%=============================================================    
    else
      % LOCATE THE MINIMUM OF THE CUBIC INTERPOLATION
      mat_temp = [1/lambda^2 -1/lambda_old^2; -lambda_old/lambda^2 lambda/lambda_old^2];
      mat_temp = mat_temp/(lambda - lambda_old);

      %vec_temp = [perfb - perfa - dperfa*lambda; perfb_old - perfa - lambda_old*dperfa];
      vec_temp = [perfb.T - perfa.T - dperfa*lambda; perfb_old.T - perfa.T - lambda_old*dperfa];
%=============================================================  
      cub_coef = mat_temp*vec_temp;
      c1 = cub_coef(1); c2 = cub_coef(2);
      disc = c2^2 - 3*c1*dperfa;
      if c1 == 0
        lambda_temp = -dperfa/(2*c2);
      else
        lambda_temp = (-c2 + sqrt(disc))/(3*c1);
      end
    
    end

    % CHECK TO SEE THAT LAMBDA DECREASES ENOUGH
  if lambda_temp > up_lim*lambda
    lambda_temp = up_lim*lambda;
  end
    
  % SAVE OLD VALUES OF LAMBDA AND FUNCTION DERIVATIVE
  lambda_old = lambda;
    perfb_old = perfb;    
    
  % CHECK TO SEE THAT LAMBDA DOES NOT DECREASE TOO MUCH
  if lambda_temp < low_lim*lambda
    lambda = low_lim*lambda;
  else
    lambda = lambda_temp;
  end
    
  % COMPUTE PERFORMANCE AND SLOPE AT NEW END POINT
    X_temp = X + lambda*dX;
    net_temp = setx(net,X_temp);

    %[perfb,E,Ac,N,Zb,Zi,Zl] = calcperf(net_temp,X_temp,Pd,Tl,Ai,Q,TS);
    [perfb,E,Ac,N,Zb,Zi,Zl] = calcperf_df(net_temp,X_temp,Pd,Tl,Ai,Q,TS,dF,ro);
%============================================================================     
    g_flag = 0;
    cnt2 = cnt2 + 1;

  end

start = 0;

end

if (g_flag == 0)

  %gX = -calcgx(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb,Q,TS);
  %gX = -calcgx_df(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb.T,Q,TS,dF);
  gX = -calcgx_df(net_temp,X_temp,Pd,Zb,Zi,Zl,N,Ac,E,perfb,Q,TS,dF,ro);
%============================================================================  
else
  gX = gX_temp;
end

a = lambda;

% CHANGE INITIAL STEP SIZE TO PREVIOUS STEP
delta=a;
if delta < net.trainParam.delta
  delta = net.trainParam.delta;
end
if tol>delta/scale_tol
  tol=delta/scale_tol;
end

retcode1 = [cnt1 cnt2 retcode];

