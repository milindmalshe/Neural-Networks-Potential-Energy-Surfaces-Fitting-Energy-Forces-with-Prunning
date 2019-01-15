function [dAP,dAN,ddAA] = CalcDfNN_Loop(net,PD,Ac,r)

%Q - Number of training data.
%dAP{m} - Derivative of NN wrt. r inputs, size (Sm x Q). r = 1,...,R.
%Ac{m} - Layer Output, size (Sm x Q).
%dAN{m} - dA{m}/dN{m}, size (Sm x Q).
%ddAA{m} - d(d(A{m}/dN{m})/dA{m}, size (Sm x Q).
%r  - Dimension of the input, a scalar.

%Description
%SAME AS CalcDfNN.m, This is the version where we use
%for-loop, to reduce the matrix size.

%================================================================

dAP = cell(net.numLayers,1);
dAN = cell(net.numLayers,1);
ddAA = cell(net.numLayers,1);

[R,Q] = size(PD{1,1});

for i=1:net.numLayers,
    
    %'tansig' is assumed at all layers, except the last layer.
    dAN{i} = (1 - (Ac{i} .* Ac{i}));
    ddAA{i} = -2*Ac{i};
    
    %Compute Derivaive of Each Layer.
    if isequal(i,1),
        
        %For Input Layer, IW is used.
        dAP{i} = dAN{i} .* (net.IW{i,i}(:,r)*ones(1,Q));
    
    elseif isequal(i,net.numLayers),
        
        %For the last layer, dA/dN is 1, size (1 x Q)
        dAN{i} = ones(net.layers{i}.size,Q);
        dAP{i} = dAN{i} .* (net.LW{i,i-1} * dAP{i-1});
        ddAA{i} = zeros(net.layers{i}.size,Q);
        
    else

        dAP{i} = dAN{i} .* (net.LW{i,i-1} * dAP{i-1});
    
    end
        
end