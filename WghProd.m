function mulW = WghProd(net,epoch,mulW)

%Parameters:
%net        - Neural network.
%mulW.df1   - sqrt((w2*w11)^2+(w2*w12)^2+...+(w2*w1R)^2).
%mulW.df2   - sqrt((w2*w11^2)^2+(w2*w12^2)^2+...+(w2*w1R^2)^2).

%======================================================================

if isequal(nargin,1),
    epoch = 1;
else
    if isequal(epoch,0),
        epoch = 1;
    else
        epoch = epoch+1;
    end
end

mulW.df(:,epoch) = abs(net.LW{2,1}').*sqrt(sum(net.IW{1,1}.^2,2));
mulW.df2(:,epoch) = 2*abs(net.LW{2,1}').*sqrt(sum(net.IW{1,1}.^4,2));
