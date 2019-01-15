function [gBdf,gIWdf,gLWdf]=calcgrad_df(net,PD,Ac,dF)

%PD - Inputs, size (R x Q).
%R - Input Dimension.
%dF.Value - Derivative Information, size (K x QR), K = No. of Neurons in the
%           last layer.
%dF.Flag - Flag matrix of the derivative information, size (K x QR).
%Edf - Derivative Error, size (K x QR) = (dF - dFhat{Last Layer}).
%dN{m} - dJ/dn{m}, size (Sm x QR).
%dNP{m} - d(dN{m})/dp, size (Sm x Q).
%dAP{m} - d(A{m})/dp, size (Sm x QR).
%dAN{m} - d(A{m})/d(N{m}), size (Sm x Q).
%ddAA{m} - d(d(A{m}/dN{m})/dA{m}, size (Sm x Q).
%Ac{m} - Layer outputs, size (Sm x Q).
%gBdf{m} - Gradient of Biases, size (Sm x 1).
%gIWdf{1,1} - Gradient of Input Weights, size (S1 x R).
%gLWdf{m,m-1} - Gradient of Layer Weights, size (Sm x Sm-1).

%Related functions:
%   - CalcDfNN
%   - trainscg_df
%   - calcgx_df

%===================================================================

%SAME AS calcgrad_df.m, Version 0. This follows the new derivation where the
%for-loop method is possible.

gBdf = cell(net.numLayers,1);
gIWdf = cell(net.numLayers,net.numInputs);
gLWdf = cell(net.numLayers,net.numLayers);
dN = cell(net.numLayers,1);
dNP = cell(net.numLayers,1);
[R,Q] = size(PD{1,1});

%To compute the number of derivative terms, for MSE calculation.
Qdf = sum(dF.Flag(:));

%Propagation to compute dAP (layer outputs wrt. NN inputs) for every layer.
ddAA_flag = 1;
[dAP,dAN,ddAA] = CalcDfNN(net,PD,Ac,ddAA_flag);

%Compute Derivative Error
Edf = (dF.Value - dAP{net.numLayers}) .* dF.Flag;

%Create the sparse matrix
C0 = sparse(reshape(repmat(1:R,Q,1),1,R*Q),1:Q*R,ones(1,Q*R),R,Q*R);
%C1 = repmat(sparse(1:Q,1:Q,ones(1,Q),Q,Q),R,1);
C1 = sparse(1:Q*R,repmat(1:Q,1,R),ones(1,R*Q),R*Q,Q);

%Backpropagation
for i=net.hint.bpLayerOrder,
    
    %Compute dN and dNP
    if net.targetConnect(i),
        
        %Param in the last layer
        dN{i} = repmat(dAN{i},1,R) .* Edf;
        dNP{i} = ddAA{i} .* ((dAP{i} .* Edf) * C1);
        
    else
        
        dN{i} = repmat(dAN{i},1,R) .* (net.LW{i+1,i}'*dN{i+1});
        
        dNP{i} = (dAN{i} .* (net.LW{i+1,i}'*dNP{i+1})) + ...
            (ddAA{i} .* ((dAP{i} .* (net.LW{i+1,i}'*dN{i+1})) * C1));
    
    end
    
    %Update Biases gBdf{m}
    gBdf{i} = -(2/Qdf)*sum(dNP{i},2);
    
    %Update Weights 
    if isequal(i,1),
        
        %Input Weights
        gIWdf{i,i} = -(2/Qdf)*(dN{i}*C0' + dNP{i}*PD{1,1}');
    
    else
        
        %Layer Weights
        gLWdf{i,i-1} = -(2/Qdf)*(dN{i}*dAP{i-1}' + dNP{i}*Ac{i-1}');
        
    end
    
end
