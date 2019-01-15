function J = calcjx_df(net,PD,Ac,dAP,dAN,ddAA,r)
%CALCJX Calculate weight and bias performance Jacobian as a single matrix.

%       J = calcjx_df(net,PD,Ac,dF)

%   Parameters
%	dN{m} -   dEdf/dn{m}, size (KQ x Sm).
%	dNP{m}-   d(dN{m})/dP, size (KQR x Sm).
     
%====================================

%INITIALIZE MATRICES
J = [];
Layers = net.numLayers;
dN = cell(Layers,1);
dNP = cell(Layers,1);
Jw = cell(Layers,1);
Jb = cell(Layers,1);

%CALCULATE Q AND R
[R,Q] = size(PD{1,1});
K = net.layers{Layers}.size;

%CHECK IF THE FOR-LOOP IS NEEDED.
if nargin < 7,
    r=0;
else
    R=1;
end
QR = Q*R;

%BACK PROPAGATATION
for i=net.hint.bpLayerOrder,
    
    %COMPUTE dN and dNP
    if net.targetConnect(i),
        
        %PARM IN THE LAST LAYER (USING SPARSE TO SAVE MEMORY)
        %dN{i} = repmat(speye(K),Q,1);
        %dNP{i} = zeros(K*QR,K);
        dN{i} = -sparse(1:K*Q,repmat(1:K,1,Q),reshape(dAN{i},1,K*Q));
        dNP{i} = -sparse(1:K*QR,repmat(1:K,1,QR),reshape(kron(ones(R,1),ddAA{i}),1,K*QR));
        
    else
        
        dN{i} = kron(dAN{i}',ones(K,1)) .* (dN{i+1}*net.LW{i+1,i});        
        dNP{i} = (kron(repmat(ddAA{i}',R,1).*dAP{i}',ones(K,1)) .* repmat(dN{i+1}*net.LW{i+1,i},R,1)) + ...
            (repmat(kron(dAN{i}',ones(K,1)),R,1) .* (dNP{i+1}*net.LW{i+1,i}));
    
    end
    
    %COMPUTE THE JACOBIAN J{m}
    Sm = net.layers{i}.size;
    if isequal(i,1),
        
        Sm_1 = net.inputs{i}.size;
        
        %SPARSE MATRIX FOR Jw COMPUTATION
        %C0 = sparse(reshape(repmat(1:R,Q,1),1,R*Q),1:Q*R,ones(1,Q*R),R,Q*R);
        %R=Sm_1; QR=QR*2;
        
        %THIS IS USED IF THE ARRANGE OF THE NETWORK PARAMETER VECTOR 
        %FOLLOWS THE TEXT BOOK.
        %Jw{i} = (kron(dNP{i},ones(1,Sm_1)) .* kron(repmat(PD{1,1}',R,Sm),ones(K,1))) + ...
		%	(kron(repmat(dN{i},R,1),ones(1,Sm_1)) .* kron(repmat(C0',1,Sm),ones(K,1)));
        
        %THIS IS USED IF THE VECTOR IS ARRANGED AS IN MATLAB TOOLBOX (FEB 09, 07).
        %Jw{i} = (repmat(dNP{i},1,Sm_1) .* kron(repmat(PD{1,1}',R,1),ones(K,Sm))) + ...
		%	(repmat(dN{i},R,Sm_1) .* kron(C0',ones(K,Sm)));
        
        %SAME AS MATLAB TOOLBOX, BUT SAVE MEMORY (FEB 25, 07)
        A = reshape(repmat(reshape(1:K*QR,K*Q,R)',1,Sm)',1,K*QR*Sm);
        B = reshape(repmat(1:R*Sm,K*Q,1),1,K*QR*Sm);
        
        if isequal(r,0),
            C1 = sparse(A,B,ones(1,QR*K*Sm),QR*K,Sm_1*Sm);
        else
            C1 = sparse(A,B+(r-1)*Sm,ones(1,QR*K*Sm),QR*K,Sm_1*Sm);
        end

        Jw{i} = (repmat(dNP{i},1,Sm_1) .* kron(repmat(PD{1,1}',R,1),ones(K,Sm))) + ...
			(repmat(dN{i},R,Sm_1) .* C1);
        
    else
        
        Sm_1 = net.layers{i-1}.size;
        
        %THIS IS USED IF THE ARRANGE OF THE NETWORK PARAMETER VECTOR 
        %FOLLOWS THE TEXT BOOK.
        %Jw{i} = (kron(dNP{i},ones(1,Sm_1)) .* kron(repmat(Ac{i-1}',R,Sm),ones(K,1))) + ...
        %  (kron(repmat(dN{i},R,1),ones(1,Sm_1)) .* kron(repmat(dAP{i-1}',1,Sm),ones(K,1)));
        
        %SAME AS VECTOR ARRANGEMENT IN MATLAB TOOLBOX (FEB 09, 07)
        Jw{i} = (repmat(dNP{i},1,Sm_1) .* kron(repmat(Ac{i-1}',R,1),ones(K,Sm))) + ...
          (repmat(dN{i},R,Sm_1) .* kron(dAP{i-1}',ones(K,Sm)));
      
    end
    Jb{i} = dNP{i};
    
    %AUGMENT FOR THE TOTAL JACOBIAN J
    if issparse(Jb{i}),
        Jb{i} = full(Jb{i});
    end
    if issparse(Jw{i}),
        Jw{i} = full(Jw{i});
    end
    J = [Jw{i} Jb{i} J];
    
end



