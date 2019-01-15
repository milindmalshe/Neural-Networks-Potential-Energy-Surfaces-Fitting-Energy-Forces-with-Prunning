function [minD,sD] = mindist_pt_weight(net,P,width_param)

%THIS FUNCTION IS TO CALCULATE THE MINIMUM DISTANCE BETWEEN A DATA POINT TO
%THE WEIGHT LINE.
%------------------------------

[R,Q] = size(P);
HWidth = atanh(sqrt(width_param))./sqrt(sum(net.IW{1,1}.^2,2));

N = net.IW{1,1}*P+repmat(net.b{1},1,Q);

D = abs(N)./repmat(sqrt((sum(net.IW{1,1}.^2,2))),1,Q);

[minD.v,minD.q] = min(D,[],2);

%Find number of data points with small distance to the weight lines.
for i=1:net.layers{1}.size,
    sD{i}.D = find(D(i,:) <= HWidth(i));
    sD{i}.num = numel(sD{i}.D);
end
