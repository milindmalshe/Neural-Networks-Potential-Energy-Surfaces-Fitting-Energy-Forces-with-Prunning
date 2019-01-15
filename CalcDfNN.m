function [dAP,dAN,ddAA] = CalcDfNN(net,PD,Ac,ddAA_flag,r)

%Q          - Number of training data.
%dAP{m}     - Derivative of NN wrt. inputs, size (Sm x QR).
%Ac{m}      - Layer Output, size (Sm x Q).
%dAN{m}     - dA{m}/dN{m}, size (Sm x Q).
%ddAA{m}    - d(d(A{m}/dN{m})/dA{m}, size (Sm x Q).
%ddAA_flag  - if set at 1, ddAA will also be computed, otherwise only
%             dAP and dAN will be calculated.
%r          - To let the code know which input variable it calculates.

%===================================================================

dAP = cell(net.numLayers,1);
dAN = cell(net.numLayers,1);
ddAA = cell(net.numLayers,1);

[R,Q] = size(PD{1,1});

%Define C0 sparse matrix for the matrix approach.
C0 = sparse(reshape(repmat(1:R,Q,1),1,R*Q),1:Q*R,ones(1,Q*R),R,Q*R);

%Check if the memory-save approach is needed.
if nargin < 5,
    r=0;
else
    R=1;
end

for i=1:net.numLayers,
    
    %'tansig' is assumed at all layers, except the last layer.
    if ~isequal(i,net.numLayers),
        dAN{i} = (1 - (Ac{i} .* Ac{i}));
        if ddAA_flag,
            ddAA{i} = -2*Ac{i};
        end
    end
    
    %Compute Derivaive of Each Layer.
    if isequal(i,1),
        
        %Check if memory-save approach is needed.
        %For Input Layer, IW is used.
        if isequal(r,0),
            dAP{i} = repmat(dAN{i},1,R) .* (net.IW{i,i} * C0);
        else
            dAP{i} = dAN{i} .* (net.IW{i,i}(:,r)*ones(1,Q));
        end

    elseif isequal(i,net.numLayers),
        
        %For the last layer, dA/dN is 1, size (1 x Q)
        dAN{i} = ones(net.layers{i}.size,Q);
        dAP{i} = repmat(dAN{i},1,R) .* (net.LW{i,i-1} * dAP{i-1});
        if ddAA_flag,
            ddAA{i} = zeros(net.layers{i}.size,Q);
        end
        
    else

        dAP{i} = repmat(dAN{i},1,R) .* (net.LW{i,i-1} * dAP{i-1});
    
    end
        
end