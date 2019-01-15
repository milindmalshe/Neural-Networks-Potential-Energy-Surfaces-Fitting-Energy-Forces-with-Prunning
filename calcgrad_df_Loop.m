function [gBdf,gIWdf,gLWdf]=calcgrad_df_Loop(net,PD,Ac,dF)

%PD - Inputs, size (R x Q).
%R - Input Dimension.
%dF - Derivative Information, size (K x QR), K = No. of Neurons in the
%     last layer.
%dfFlag - Flag matrix of the derivative information, size (K x QR).
%Edf - Derivative Error, size (K x Q) = (dF - dFhat{Last Layer}).
%dN{m} - dJ/dn{m}, size (Sm x Q).
%dNP{m} - d(dN{m})/dp, size (Sm x Q).
%dAP{m} - d(A{m})/dp, size (Sm x Q).
%dAN{m} - d(A{m})/d(N{m}), size (Sm x Q).
%ddAA{m} - d(d(A{m}/dN{m})/dA{m}, size (Sm x Q).
%Ac{m} - Layer outputs, size (Sm x Q).
%gBdf{m} - Gradient of Biases, size (Sm x 1).
%gIWdf{1,1} - Gradient of Input Weights, size (S1 x R).
%gLWdf{m,m-1} - Gradient of Layer Weights, size (Sm x Sm-1).

%Description
%SAME AS calcgrad_df.m, Version 1. This is the version where we use
%for-loop, to reduce the matrix size.

%Related functions:
%   - CalcDfNN_Loop
%   - trainscg_df
%   - calcgx_df

%===================================================================

gBdf = cell(net.numLayers,1);
gIWdf = cell(net.numLayers,net.numInputs);
gLWdf = cell(net.numLayers,net.numLayers);
dN = cell(net.numLayers,1);
dNP = cell(net.numLayers,1);
[R,Q] = size(PD{1,1});

%For-loop over Input dimension
for r = 1:R,
    
    %Propagation to compute dAP (layer outputs wrt. NN inputs) for every layer.
    [dAP,dAN,ddAA] = CalcDfNN_Loop(net,PD,Ac,r);
    
    for i=net.hint.bpLayerOrder,
    
        %Initialize Matrices
        dN{i} = zeros(net.layers{i}.size,Q);
        dNP{i} = zeros(net.layers{i}.size,Q);
        if isequal(r,1),
            gBdf{i} = zeros(size(net.b{i}));
            if isequal(i,1),
                gIWdf{i,i} = zeros(size(net.IW{i,i}));
            else
                gLWdf{i,i-1} = zeros(size(net.LW{i,i-1}));
            end
        end

        %Compute Derivative Error wrt. each input dimension
        Edf = (dF.Value(:,(r-1)*Q+1:r*Q) - dAP{net.numLayers}).*dF.Flag(:,(r-1)*Q+1:r*Q);

        %Compute dN and dNP
        if net.targetConnect(i),
           
            dN{i} = dAN{i} .* Edf;
            dNP{i} = ddAA{i} .* (dAP{i} .* Edf);
        
        else

            dN{i} = dAN{i} .* (net.LW{i+1,i}'*dN{i+1});
            dNP{i} = (dAN{i} .* (net.LW{i+1,i}'*dNP{i+1})) + ...
                (ddAA{i} .* (dAP{i} .* (net.LW{i+1,i}'*dN{i+1})));
    
        end

        %Update Biases gBdf{m}
        gBdf{i} = (-2/Q)*sum(dNP{i},2) + gBdf{i};
    
        %Update Weights 
        if isequal(i,1),
            
            %Constant Matrix
            C = zeros(1,R);
            C(r) = 1;
        
            %Input Weights
            gIWdf{i,i} = (-2/Q)*(sum(dN{i},2)*C + dNP{i}*PD{1,1}') + gIWdf{i,i};
    
        else
    
            %Layer Weights
            gLWdf{i,i-1} = (-2/Q)*(dN{i}*dAP{i-1}' + dNP{i}*Ac{i-1}') + gLWdf{i,i-1};
        
        end
        
    end
    
end
    
