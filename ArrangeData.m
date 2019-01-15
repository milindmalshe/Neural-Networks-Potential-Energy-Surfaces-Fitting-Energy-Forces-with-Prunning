function [TR,VV] = ArrangeData(trnFcn,TR,VD,doValidation)

%THIS FUNCTION IS TO ARRANGE DATA SO THAT IT WILL SUIT FOR TRAINING
%NEURAL NETWORKS WITH DERIVATIVE INFORMATION.

%-------------------------------------------------------------

[SM,Qtr] = size(TR.T);

if isequal(trnFcn,'bfg_df') || isequal(trnFcn,'lm_df'),
    %If TRAINBFG_DF, Validation can be included into training set, if
    %needed.
    if ~doValidation,
        
        if ~isempty(VD.P),
            [R,Qtr] = size(TR.P);
            [R,Qvd] = size(VD.P);
            TR.P = [TR.P VD.P];
            TR.T = [TR.T VD.T];
            TR.Df2 = [TR.Df2 VD.Df2];
            for k = 1:SM,
                DfTemp = reshape(TR.Df(k,:),Qtr,R)';
                DfTemp1 = reshape(VD.Df(k,:),Qvd,R)';
                Df = [DfTemp DfTemp1];
                TR_temp.Df(k,:) = reshape(Df',1,(Qtr+Qvd)*R);
                
                MapTemp = reshape(TR.DfMap(k,:),Qtr,R)';
                MapTemp1 = reshape(VD.DfMap(k,:),Qvd,R)';
                Map = [MapTemp MapTemp1];
                TR_temp.DfMap(k,:) = reshape(Map',1,(Qtr+Qvd)*R);
            end
            TR.Df = TR_temp.Df;
            TR.DfMap = TR_temp.DfMap;
        end

        VV.DfvdFlag = 0;    %This parameter will empty 'VV' in 'trainbfg_df'.
        VV.P = TR.P;
        VV.T = TR.T;
        VV.Dfvd = TR.Df;
        VV.DfvdMap = TR.DfMap;
        
    else

        [R,Q] = size(TR.P);
        
        VV.DfvdFlag = 1;    
        VV.P = VD.P;
        VV.T = VD.T;
        [R,Qvd] = size(VD.P);
        VV.Dfvd = VD.Df;
        VV.DfvdMap = VD.DfMap;
        
    end

    VV.Dftr = TR.Df;
    VV.DftrMap = TR.DfMap;
    SeenDf = TR.Df.*TR.DfMap;
    VV.SqMaxRatio = (max(abs(SeenDf(:)))/max(abs(TR.T(:))))^2;
    
else

    [R,Q] = size(TR.P);
    VV.Dftr = TR.Df;
    VV.DftrMap = TR.DfMap;
   
    if ~doValidation,
        
        if ~isempty(VD.P),
            [R,Qtr] = size(TR.P);
            [R,Qvd] = size(VD.P);
            for k = 1:SM,
                DfTemp = reshape(TR.Df(k,:),Qtr,R)';
                DfTemp1 = reshape(VD.Df(k,:),Qvd,R)';
                Df = [DfTemp DfTemp1];
                TR_New.Df(k,:) = reshape(Df',1,(Qtr+Qvd)*R);
                
                MapTemp = reshape(TR.DfMap(k,:),Qtr,R)';
                MapTemp1 = reshape(VD.DfMap(k,:),Qvd,R)';
                Map = [MapTemp MapTemp1];
                TR_New.DfMap(k,:) = reshape(Map',1,(Qtr+Qvd)*R);
            end
            TR_New.P = [TR.P VD.P];
            TR_New.T = [TR.T VD.T];
            TR_New.Df2 = [TR.Df2 VD.Df2];
            TR = TR_New;

        end

        VV = [];
        
    else
        
        VV.DfvdFlag = 1;    
        VV.P = VD.P;
        VV.T = VD.T;
        [R,Qvd] = size(VD.P);
        VV.Dfvd = VD.Df;
        VV.DfvdMap = VD.DfMap;
        SeenDf = TR.Df.*TR.DfMap;
        VV.SqMaxRatio = (max(abs(SeenDf))/max(abs(TR.T(:))))^2;
        
    end
    
end
