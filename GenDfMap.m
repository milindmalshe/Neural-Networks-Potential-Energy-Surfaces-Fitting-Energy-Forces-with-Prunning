function TR = GenDfMap(TR,fracQ,fracR);

%Description:
%This function is used to randomly take off some of the derivative
%--------------------------------------------------

[R,Q] = size(TR.P);
[SM,Q] = size(TR.T);
TR.DfMap = ones(SM,Q*R);

if isequal(nargin,1),
    
    %Check if NaN appears in the derivative information.
    %If yes, set NaN to very large negative value.
    
    TR.DfMap = ~isnan(TR.Df);
    nan = find(TR.DfMap == 0);
    TR.Df(nan) = -1e6;
    
elseif isequal(nargin,2),
       
    %This will take off just only some data points.
    %Every data still has R-derivative information.
   
    numOffQ = ceil(fracQ*Q);
    indexQ = ceil((Q-1)*rand(numOffQ,1));
    indexQ = unique(indexQ);
    
    for r=1:R,
        
        TR.DfMap(:,indexQ + (r-1)*Q) = 0;
    
    end
    
else
    
    %The derivative information for some data shall have
    %less than R derivative.
    
end

    

