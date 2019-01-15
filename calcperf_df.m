function [perf,E,Ac,N,BZ,IWZ,LWZ]=calcperf_df(net,X,PD,T,Ai,Q,TS,dF,ro)
%CALCPERF Calculate network outputs, signals, and performance.
%
%=================================================
%======================================================================

% Concurrent biases
BZ = cell(net.numLayers,1);
ones1xQ = ones(1,Q);
for i=net.hint.biasConnectTo
  BZ{i} = net.b{i}(:,ones1xQ);
end

% Signals
IWZ = cell(net.numLayers,net.numInputs,TS);
LWZ = cell(net.numLayers,net.numLayers,TS);
Ac = [Ai cell(net.numLayers,TS)];
N = cell(net.numLayers,TS);

% Shortcuts
numLayerDelays = net.numLayerDelays;
inputConnectFrom = net.hint.inputConnectFrom;
layerConnectFrom = net.hint.layerConnectFrom;
biasConnectFrom = net.hint.biasConnectFrom;
inputWeightFcn = net.hint.inputWeightFcn;
layerWeightFcn = net.hint.layerWeightFcn;
netInputFcn = net.hint.netInputFcn;
transferFcn = net.hint.transferFcn;
layerDelays = net.hint.layerDelays;
IW = net.IW;
LW = net.LW;

% Simulation
for ts=1:TS
  for i=net.hint.simLayerOrder
  
    ts2 = numLayerDelays + ts;
  
    % Input Weights -> Weighed Inputs
  inputInds = inputConnectFrom{i};
    for j=inputInds
    switch inputWeightFcn{i,j}
    case 'dotprod'
      IWZ{i,j,ts} = IW{i,j} * PD{i,j,ts};
    otherwise
        IWZ{i,j,ts} = feval(inputWeightFcn{i,j},IW{i,j},PD{i,j,ts});
    end
    end
    
    % Layer Weights -> Weighted Layer Outputs
  layerInds = layerConnectFrom{i};
    for j=layerInds
    thisLayerDelays = layerDelays{i,j};
    if (length(thisLayerDelays) == 1) & (thisLayerDelays == 0)
      Ad = Ac{j,ts2};
    else
      Ad = cell2mat(Ac(j,ts2-layerDelays{i,j})');
    end
    switch layerWeightFcn{i,j}
    case 'dotprod'
        LWZ{i,j,ts} = LW{i,j} * Ad;
    otherwise
        LWZ{i,j,ts} = feval(layerWeightFcn{i,j},LW{i,j},Ad);
    end
    end
  
    % Net Input Function -> Net Input
  if net.biasConnect(i)
      Z = [IWZ(i,inputInds,ts) LWZ(i,layerInds,ts) BZ(i)];
  else
      Z = [IWZ(i,inputInds,ts) LWZ(i,layerInds,ts)];
  end
  switch netInputFcn{i}
  case 'netsum'
      N{i,ts} = Z{1};
      for k=2:length(Z)
        N{i,ts} = N{i,ts} + Z{k};
      end
  case 'netprod'
      N{i,ts} = Z{1};
      for k=2:length(Z)
        N{i,ts} = N{i,ts} .* Z{k};
      end
  otherwise
      N{i,ts} = feval(netInputFcn{i},Z{:});
    end
  
    % Transfer Function -> Layer Output
  switch transferFcn{i}
  case 'purelin'
    Ac{i,ts2} = N{i,ts};
  case 'tansig'
    n = N{i,ts};
    a = 2 ./ (1 + exp(-2*n)) - 1;
      k = find(~finite(a));
      a(k) = sign(n(k));
      Ac{i,ts2} = a;
  case 'logsig'
      n = N{i,ts};
      a = 1 ./ (1 + exp(-n));
      k = find(~finite(a));
      a(k) = sign(n(k));
    Ac{i,ts2} = a;
  otherwise
      Ac{i,ts2} = feval(transferFcn{i},N{i,ts});
  end
  end
end

% CALCE: E = calce(net,Ac,T,TS);
%===============================

E = cell(net.numLayers,TS);

for ts = 1:TS
  for i=net.hint.targetInd
    E{i,ts} = T{i,ts} - Ac{i,ts+numLayerDelays};
  end
end

% Performance
%============

performFcn = net.performFcn;
if length(performFcn) ==0
  performFcn = 'nullpf';
end

%perf = feval(performFcn,E,X,net.performParam);
perf_f = feval(performFcn,E,X,net.performParam);
%==================================================


%Add the derivative MSE
ddAA_flag = 0;
dAP = CalcDfNN(net,PD,Ac,ddAA_flag);
perf_df = sse((dF.Value-dAP{net.numLayers}).*dF.Flag)/sum(dF.Flag(:));
c1 = 1;
%c2 = 1e4*(1/dF.SqMaxRatio); %Old version ro
c2 = ro;
%=================================================================

perf.T = c1*perf_f + c2*perf_df;
perf.F = perf_f;
perf.Df = perf_df;
%==================================================
