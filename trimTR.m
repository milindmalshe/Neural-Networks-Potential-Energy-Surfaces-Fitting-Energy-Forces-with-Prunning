function TR_trim = trimTR(TR,func_no)

%DESCRIPTION
%This function is to cut down the size of the training data of the
%functions in the simple plan so that it will be on the same region as the
%testing data. This is to eliminate the problem when the errors at the end
%points were much higher.

%------------------------------------------------------------------------

TR_trim = TR;

if isequal(func_no,1) || isequal(func_no,2) || isequal(func_no,3), %1D Functions
    
    if isequal(func_no,1) || isequal(func_no,3),
    
        xx = find(TR.P >= -0.8 & TR.P <= 0.8); 

    elseif isequal(func_no,2),
    
        xx = find(TR.P >= 0.1 & TR.P <= 0.9); 

    end
    TR_trim.P = TR.P(xx);
    TR_trim.T = TR.T(xx);
    TR_trim.Df = TR.Df(xx);
    TR_trim.Df2 = TR.Df2(xx);
    TR_trim.DfMap = TR.DfMap(xx);

elseif isequal(func_no,4), %2D sinc
    
    xx = find(abs(TR.P(1,:)) <= 0.8 & abs(TR.P(1,:)) <= 0.8);
    
    TR_trim.P = TR.P(:,xx);
    TR_trim.T = TR.T(xx);
    TRDF_temp = reshape(TR.Df,numel(TR.Df)/2,2)';
    TRDFMAP_temp = reshape(TR.DfMap,numel(TR.DfMap)/2,2)';
    TRDF_temp = TRDF_temp(:,xx);
    TRDFMAP_temp = TRDFMAP_temp(:,xx);
    TR_trim.Df = reshape(TRDF_temp',1,numel(TRDF_temp));
    TR_trim.DfMap = reshape(TRDFMAP_temp',1,numel(TRDFMAP_temp));

end

%FOR MD Functions, TR_trim = TR;
