%THIS FILE IS TO COMPUTE UN-NORMALIZED 2nd DERIVATIVE ERRORS FOR 2D Sinc.

clear;

load_dirData = 'Z:\Neuron Removal\Sinc\DataSet 2\Original\';
load_dir = 'Z:\Neuron Removal\Sinc\DataSet 2\xDf_I\Df_closept\Thrd_mxmndist\0.1Ratio\Trainlm_df\From 150Kepochs\';
save_dir = 'Z:\Neuron Removal\Sinc\DataSet 2\xDf_I\Df_closept\Thrd_mxmndist\0.1Ratio\Trainlm_df\From 150Kepochs\';
Ext_Load = '';
Ext_Save = '';

M=10;
trnFcn = 'lm_df';

%Load Data
load([load_dirData 'Data_f4' Ext_Load '']);

RMSE_TR_2nd = []; RMSE_TS_2nd = [];

for j=1:10,
    
    TRN=TR{j};
    
    %Load network
    filename_LL = [trnFcn num2str(j) '_f3' Ext_Load];
    load([load_dir filename_LL]);

    %Reduced training set
    xx = find(abs(TRN.P(1,:))<0.8 & abs(TRN.P(2,:))<0.8);
    TRN_trim.P = TRN.P(:,xx);

    %Normalize the input data (only testing, since training input is already normalized)
    TS_N.P = tramnmx(TS.P,TRN.minp,TRN.maxp);
    
    %Compute the normalized 2nd Derivatives of the network (Input is
    %normalized)
    d2Ap_TR = Calc2dAP(net,TRN_trim.P);
    d2Ap_TS = Calc2dAP(net,TS_N.P);
            
    [R,Qtr]=size(TRN_trim.P);
    [R,Qts]=size(TS_N.P);

    %Unnormalize the 2nd derivatives
    d2Ap_TR_UN = postmnmxDf2(d2Ap_TR,TRN.minp,TRN.maxp,TRN.mint,TRN.maxt,R,Qtr);
    d2Ap_TS_UN = postmnmxDf2(d2Ap_TS,TRN.minp,TRN.maxp,TRN.mint,TRN.maxt,R,Qts);
    
    %Unnormalize the input data (only for training, since testing is unnormalized)
    TRN_UN.P = postmnmx(TRN_trim.P,TRN.minp,TRN.maxp);
    
    %Get the true 2nd derivatives of the sinc function
    [T_TR,Df_TR,Df2_TR] = SincFunc(TRN_UN.P);
    [T_TS,Df_TS,Df2_TS] = SincFunc(TS.P);
    
    %Compute and collect errors
    RMSE_TR_2nd = [RMSE_TR_2nd sqrt(mse(Df2_TR-d2Ap_TR_UN))];
    RMSE_TS_2nd = [RMSE_TS_2nd sqrt(mse(Df2_TS-d2Ap_TS_UN))];

end

save([save_dir 'Error_2nd' Ext_Load],'RMSE_TR_2nd','RMSE_TS_2nd');

