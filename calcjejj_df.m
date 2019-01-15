function [JE_df,JJ_df,normJE] = calcjejj_df(net,PD,Ac,dF,mem_reduc,MR_R)
%CALCJEJJ Calculate Jacobian performance vector.
%=========================================

%CALCULATE Q AND R
[R,Q] = size(PD{1,1});
[K,QR] = size(dF.Flag);

%DEFINE ddAA_flag variable
ddAA_flag = 1;

if 1 %MTH2_9_10 isequal(mem_reduc,1),
	
	if ~MR_R

		%COMPUTE dA/dP, dA/dN AND d(dA/dN)/dA.
		[dAP,dAN,ddAA] = CalcDfNN(net,PD,Ac,ddAA_flag);

		%COMPUTE THE DERIVATIVE ERRORS
		Edf = reshape((dF.Value - dAP{net.numLayers}) .* dF.Flag,K*QR,1);

		%COMPUTE THE JACOBIAN MATRIX
		J = calcjx_df(net,PD,Ac,dAP,dAN,ddAA);

		%COMPUTE J'*E AND J'*J
		JE_df = J'*Edf;
		JJ_df = J'*J;

	else
	
		for r=1:R,
		
			%COMPUTE dA/dPr.
            [dAP,dAN,ddAA] = CalcDfNN(net,PD,Ac,ddAA_flag,r);
		
			%COMPUTE THE DERIVATIVE ERRORS WITH RESPECT TO EACH INPUT DIMENSION
			Edf_t = (dF.Value(:,(r-1)*Q+1:r*Q) - dAP{net.numLayers}).*dF.Flag(:,(r-1)*Q+1:r*Q);

			%COMPUTE THE JACOBIAN MATRIX Jr
			Jr = calcjx_df(net,PD,Ac,dAP,dAN,ddAA,r);

			%COMPUTE THE TOTAL JE_df AND JJ_df
			if isequal(r,1),
				JE_df = Jr'*Edf_t';
				JJ_df = Jr'*Jr;
			else
				JE_df = JE_df + Jr'*Edf_t';
				JJ_df = JJ_df + Jr'*Jr;
			end

		end
	
	end
	
else

end

%COMPUTE THE NORM
normJE = sqrt(JE_df'*JE_df); 