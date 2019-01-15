function Df2 = postmnmxDf2(Df2N,minp,maxp,mint,maxt,R,Q);

%This function is to unnormalize the second derivative.
%Size of Df2N is 1xQ for 1D, 3xQ for 2D.


dAuAn = 0.5*(maxt-mint)*ones(1,Q);
dPnPu = (2./(maxp-minp))*ones(1,Q);

if isequal(R,1),
    
    
    
elseif isequal(R,2),
    
    %pi-pi
    Df2(1,:) = dAuAn.*Df2N(1,:).*(dPnPu(1,:).^2);
    Df2(2,:) = dAuAn.*Df2N(2,:).*(dPnPu(2,:).^2);
    
    %p1-p2
    Df2(3,:) = dAuAn.*Df2N(3,:).*(dPnPu(1,:).*dPnPu(2,:));       
    
elseif isequal(R,3),
    
    %pi-pi
    Df2(1,:) = dAuAn.*Df2N(1,:).*(dPnPu(1,:).^2);
    Df2(2,:) = dAuAn.*Df2N(2,:).*(dPnPu(2,:).^2);
    Df2(3,:) = dAuAn.*Df2N(3,:).*(dPnPu(3,:).^2);  
    
    %p1-p2
    Df2(4,:) = dAuAn.*Df2N(4,:).*(dPnPu(1,:).*dPnPu(2,:)); 
    
    %p1-p3
    Df2(5,:) = dAuAn.*Df2N(5,:).*(dPnPu(1,:).*dPnPu(3,:)); 
    
    %p2-p3
    Df2(6,:) = dAuAn.*Df2N(6,:).*(dPnPu(2,:).*dPnPu(3,:)); 
   
    
end