function Ac = CalcAc(net,PD)

%Same as CalcAc function in 'Derivative Check' Folder.


ptr = PD{1,1};

for i=1:net.numLayers,
    if isequal(i,1),
        Ac{i} = tansig(net.IW{1,1}*ptr,net.b{1});
    elseif isequal(i,net.numLayers)
        Ac{i} = purelin(net.LW{i,i-1}*Ac{i-1},net.b{i});
    else
        Ac{i} = tansig(net.LW{i,i-1}*Ac{i-1},net.b{i});
    end
end