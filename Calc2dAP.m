function D2Ap = Calc2dAP(net,P)

%Very simple code ONLY for a 2-layer net, with tansig and linear.
%   Add the case where R > 3.

PD{1,1} = P;

S1 = net.layers{1}.size;
[R,Q] = size(P);
R = net.inputs{1}.size;

Ac = CalcAc(net,PD);

if isequal(R,1),
    
    D2Ap = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}).^2,1,Q).*Ac{1}.*(1-Ac{1}.^2));
    
elseif isequal(R,2),
    
    %pi-pi
    D2Ap(1,:) = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}(:,1)).^2,1,Q).*Ac{1}.*(1-Ac{1}.^2));
    D2Ap(2,:) = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}(:,2)).^2,1,Q).*Ac{1}.*(1-Ac{1}.^2));
    
    %p1-p2
    D2Ap(3,:) = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}(:,1).*net.IW{1,1}(:,2)),1,Q).*Ac{1}.*(1-Ac{1}.^2));

elseif isequal(R,3),
    
    %pi-pi
    D2Ap(1,:) = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}(:,1)).^2,1,Q).*Ac{1}.*(1-Ac{1}.^2));
    D2Ap(2,:) = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}(:,2)).^2,1,Q).*Ac{1}.*(1-Ac{1}.^2));
    D2Ap(3,:) = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}(:,3)).^2,1,Q).*Ac{1}.*(1-Ac{1}.^2));
    
    %p1-p2
    D2Ap(4,:) = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}(:,1).*net.IW{1,1}(:,2)),1,Q).*Ac{1}.*(1-Ac{1}.^2));
    
    %p1-p3
    D2Ap(5,:) = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}(:,1).*net.IW{1,1}(:,3)),1,Q).*Ac{1}.*(1-Ac{1}.^2));
    
    %p2-p3
    D2Ap(6,:) = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}(:,2).*net.IW{1,1}(:,3)),1,Q).*Ac{1}.*(1-Ac{1}.^2));
   
else
    
    %If R > 3, compute only the 2nd derivative of pi-pi
    D2Ap = zeros(R,Q);
    for i=1:R,
        D2Ap(i,:) = -2*sum(repmat(net.LW{2,1}'.*(net.IW{1,1}(:,i)).^2,1,Q).*Ac{1}.*(1-Ac{1}.^2));
    end
    
end
