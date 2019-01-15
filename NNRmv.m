function [cdd3,cdd_s] = NNRmv(net,Data,Thrd)

%THIS FUNCTION IS TO CHECK AND OUTPUT NEURON CANDIDATES TO BE REMOVED. THE
%CANDIDATES ARE DECIDED FROM THE WEIGHT PRODUCT, THE ANGLE BETWEEN NEURONS,
%AND THE CANCELLATION OCCURS.
%======================================================================


mulW = WghProd(net);
[minD,sD] = mindist_pt_weight(net,Data.P,Thrd.width_param);

%Compute the threshold for weight product
for i=1:net.layers{1}.size,
    Thrd.df(i,:) = Thrd.df_param*Data.Df(minD.q(i));
    num_sD(i,:) = sD{i}.num; %Change varibale for first-layer check
end

%Initialize
cdd3.n = []; cdd3.r = 0; cdd3.d=[];

%FIRST CHECK
%I. Find large weight products
cdd1 = find(mulW.df >= Thrd.df);
%II. Find large first-layer weights
if Thrd.numdata*numel(Data.Df) < 1,
    cddx = find(num_sD <= 1);
else
    cddx = find(num_sD <= Thrd.numdata*numel(Data.Df));
end
cdd1 = union(cdd1,cddx);

%If non-empty, pass to neighbor and contribution checks
if ~isempty(cdd1),
    [cdd3.n,cdd3.d] = location(net,Data,cdd1,Thrd,mulW,sD);
end

%Compute the network response from the neurons in the set cdd_s
%- First, change the variable
ptr = Data.P;

%- Set the neurons not the set to be zero
netn = net;
nn = setdiff(1:net.layers{1}.size,cdd3.n);
netn.IW{1,1}(nn,:)=0; netn.LW{2,1}(nn)=0; netn.b{1}(nn)=0; netn.b{2}=0;

%- Compute the response for these neurons
cdd3.r = mean(sim(netn,ptr));

%Report the data points inside the widths
cdd3.d = unique(cdd3.d);

%=====================================================================
function [cdd3,sDU] = location(net,Data,cdd1,Thrd,mulW,sD)

%SECOND CHECK: neighbor search

%Initialize output
cdd3 = []; sDU = [];

cdd1_temp = [1:net.layers{1}.size]';
for i=1:length(cdd1),
    
    if isempty(intersect(cdd3,cdd1(i))),
        
        %- See if there are neurons' centers close to itself.
        %- Select the neurons out of the candiate list, excluding itself
        %- If no neuron is close to this neuron, compute the derivative response
        %  just from this neuron.
        remain = setdiff(cdd1_temp,cdd1(i));
    
        if ~isempty(remain),
        
            if isempty(intersect(cdd1(i),cdd3)),
                
                %Initialize
                cdd2 = [];
            
                if isequal(net.inputs{1}.size,1),
        
                    %Compute neuron's location for univariate.
                    a = -net.b{1}(cdd1(i))./net.IW{1,1}(cdd1(i),:);
                    b = -net.b{1}(remain)*ones(1,net.inputs{1}.size)./net.IW{1,1}(remain,:);
                    Dist = dist(b,a');
                    indx_temp = find(Dist <= Thrd.dist);
                    cdd2 = remain(indx_temp);
    
                else
                
                    %------------ANGLE----------------
                    %Compute neuron's angles for multivariates.
                    a = net.IW{1,1}(cdd1(i),:);
                    b = net.IW{1,1}(remain,:);
                    Dist_angle = 180*acos(b*a'./(norm(a)*sqrt(sum(b.^2,2))))/pi;
                    indx_temp_1 = find(Dist_angle <= Thrd.angle);
                    indx_temp_2 = find(180-Dist_angle <= Thrd.angle);
                    cdd2_temp_1 = remain(indx_temp_1);
                    cdd2_temp_2 = remain(indx_temp_2);
                    %---------------------------------
                    
                    if ~isempty(cdd2_temp_1),
                        a = -net.b{1}(cdd1(i))/sqrt(sum(net.IW{1,1}(cdd1(i),:).^2));
                        b = -net.b{1}(cdd2_temp_1)./sqrt(sum(net.IW{1,1}(cdd2_temp_1,:).^2,2));  
                        Dist = dist(b,a');
                        indx_temp = find(Dist <= Thrd.dist);
                        cdd2 = cdd2_temp_1(indx_temp);
                    end
                    
                    if ~isempty(cdd2_temp_2),
                        a = -net.b{1}(cdd1(i))/sqrt(sum(net.IW{1,1}(cdd1(i),:).^2));
                        b = -net.b{1}(cdd2_temp_2)./sqrt(sum(net.IW{1,1}(cdd2_temp_2,:).^2,2));
                        Dist = dist(-b,a'); %THE DIFFERENCE FROM cdd2_temp_1
                        indx_temp = find(Dist <= Thrd.dist);
                        cdd2 = [cdd2; cdd2_temp_2(indx_temp)];
                    end
                    
                end
    
                %Pass to the contribution check
                [cdd3_temp sDU_temp] = RespCancel(net,Data,cdd1(i),cdd2,Thrd,mulW,sD);
                sDU = [sDU sDU_temp];
                cdd3 = [cdd3; cdd3_temp];
                cdd1_temp = setdiff(cdd1_temp,cdd3);
            
            end
        end
    end
end
    

%=====================================================================
function [cdd3,sDU] = RespCancel(net,Data,cdd_j,cdd2,Thrd,mulW,sD)

%THIRD CHECK: contribution

%Initialize output
cdd3 = []; sDU = [];

%Change variable type
PD{1,1} = Data.P;

%Search for response cancellation through combinations of cdd_j and cdd2
cdd = [cdd_j; cdd2];
if length(cdd) > 12,
    error('Too many combinations. Please reduce the location threshold.');
end

flag = 0;
for i=length(cdd):-1:1,
    
    %Select a combination of 'i'
    if isequal(i,1),
        c = cdd_j;
    else
        c2 = combntns(cdd2,i-1);
        [num_comb,k] = size(c2);
        c = [cdd_j*ones(num_comb,1) c2];
    end
    [num_comb,k] = size(c);
        
    %Compute the derivative response from these combinations
    for j=1:num_comb,
        
        
        %Compute the total derivative from the combinadic
        netn_j = net;
        nn_j = setdiff(1:net.layers{1}.size,c(j,:));
        netn_j.IW{1,1}(nn_j,:)=0; netn_j.LW{2,1}(nn_j)=0; netn_j.b{1}(nn_j)=0; netn_j.b{2}=0;
        Ac_j = CalcAc(netn_j,PD);
        dAP_j = CalcDfNN(netn_j,PD,Ac_j,0);
        
        %1). Compute the maximum magnitude of dAP (overall response of all
        %combinadic), and indx_j indicates which point in the training set.
        dAdP_j = reshape(dAP_j{2},length(dAP_j{2})/net.inputs{1}.size,net.inputs{1}.size)';
        dAdP_Mag_j = sqrt(sum(dAdP_j.^2,1));
        [dAdP_Mag_j_mx,indx_j] = max(dAdP_Mag_j);
        %2). Compare 1) with the actual derivative value of indx_j.
        NNresp_j = dAdP_Mag_j_mx/Data.Df(indx_j);
        
        %Compute the magnitude of the derivative response - Just for reference
        dAdP_Resp_j = sqrt(mse(dAdP_Mag_j));
        
        %Check if the NNresp less than threshold
        if NNresp_j <= Thrd.Resp,
            
            flag = 1; %set flag
            cdd3 = c(j,:)';
            for k=1:length(cdd3),
                sDU = union(sDU,sD{cdd3(k)}.D);
            end

            fprintf('Remove neurons: %s',num2str(c(j,:)));
            fprintf(', Percent Df: %f',100*NNresp_j);
            fprintf(', Actual Df: %f',Data.Df(indx_j));
            fprintf(', # of Data: %g',numel(sDU));
            fprintf(', Combinadic Resp: %e',dAdP_Resp_j);
            fprintf('\n');

        end
        
        if isequal(flag,1),
            break
        end
        
    end
    
    if isequal(flag,1),
        break
    end
      
end

    
