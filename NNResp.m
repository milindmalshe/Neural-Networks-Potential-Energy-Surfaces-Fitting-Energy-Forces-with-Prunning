function [Atst,dAtst,RMSE] = NNResp(net,Data);

%--------------------------------------------------

Atst = [];
RMSE.F = [];
RMSE.Df = [];
dAtst = [];
if isempty(Data.P),
    return;
end

if ~iscell(Data.P),
    p = Data.P;
    input{1,1} = p;
    clear p;
end

[R,Q] = size(Data.P);
Df = reshape(Data.Df,Q,R)';
DfMap = reshape(Data.DfMap,Q,R)';

if (R >= 3) && (Q >= 50000),
    
    %Memory save
    M = 20;
    Err.F = []; Err.Df = []; ATST = []; DATST=[];
    for q=1:M,
        
        %Create a smaller subset
        subindex = (1+round((q-1)*Q/M)):round(q*Q/M);
        
        %For input, target, derivatives
        Pq{1,1} = input{1,1}(:,subindex);
        Tq = Data.T(:,subindex);
        Dfq = Df(:,subindex);
        Dfq = reshape(Dfq',1,numel(Dfq));
        DfMapq = DfMap(:,subindex);
        DfMapq = reshape(DfMapq',1,numel(DfMapq));
        
        Atst = CalcAc(net,Pq);
        Err.F = [Err.F Tq - Atst{net.numLayers}];  
        if ~isempty(Data.Df),
            ddAA_flag = 0;
            dAtst = CalcDfNN(net,Pq,Atst,ddAA_flag);
            Err.Df = [Err.Df ((Dfq - dAtst{net.numLayers}).*DfMapq)];
        end
        
        ATST = [ATST Atst{2}];
        DATST = [DATST reshape(dAtst{2},length(Tq),R)'];
        
    end
    RMSE.F = sqrt(mse(Err.F));
    RMSE.Df = sqrt(sse(Err.Df)/sum(Data.DfMap(:)));
    clear Atst dAtst;
    Atst{2} = ATST;
    dAtst{2} = reshape(DATST',1,R*Q);
    
else
    
    Atst = CalcAc(net,input);
    RMSE.F = sqrt(mse(Data.T - Atst{net.numLayers}));

    if ~isempty(Data.Df),
        ddAA_flag = 0;
        dAtst = CalcDfNN(net,input,Atst,ddAA_flag);
        RMSE.Df = sqrt(sse((Data.Df - dAtst{net.numLayers}).*Data.DfMap)/sum(Data.DfMap(:)));
    end

end

