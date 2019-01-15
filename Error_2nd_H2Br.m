%THIS FILE IS TO COMPUTE UN-NORMALIZED 2nd DERIVATIVE ERRORS FOR H2Br.

clear;

load_dirData = 'D:\Ohm\PhD\Work\Derivative Work\MD Data_Results\H2Br_Trajectory\Distance_MorePointsAtLargeForce\375TR points\';
load_dir = 'Z:\Neuron Removal\H2Br\375TR\xDf_I\Df_closept\Thrd_mxmndist\0.1Ratio\Trainbfg_df\From 90Kepochs\';
save_dir = 'Z:\Neuron Removal\H2Br\375TR\xDf_I\Df_closept\Thrd_mxmndist\0.1Ratio\Trainbfg_df\From 90Kepochs\';
Ext_Load = '';
Ext_Save = '';

M=10;
trnFcn = 'bfg_df';

%Load Data
load([load_dirData 'Data_f7' Ext_Load],'TR','TS_UN');

RMSE_TR_2nd = []; RMSE_TS_2nd = [];

for j=1:10,
    
    TRN=TR{j};
    [R,Qtr]=size(TRN.P);
    
    %Load network
    filename_LL = [trnFcn num2str(j) '_f7' Ext_Load];
    load([load_dir filename_LL]);

    %Normalize the input data (only testing, since training input is already normalized)
    TS_N.P = tramnmx(TS_UN{j}.P,TRN.minp,TRN.maxp);
    
    %Divide Test Set
    [R,Qts] = size(TS_N.P);
    TS_N1.P = TS_N.P(:,1:100000);
    TS_N2.P = TS_N.P(:,100001:200000);
    TS_N3.P = TS_N.P(:,200001:300000);
    TS_N4.P = TS_N.P(:,300001:400000);
    TS_N5.P = TS_N.P(:,400001:Qts);
    
    %Compute the normalized 2nd Derivatives of the network (Input is
    %normalized)
    d2Ap_TR = Calc2dAP(net,TRN.P);
    d2Ap_TS1 = Calc2dAP(net,TS_N1.P);
    d2Ap_TS2 = Calc2dAP(net,TS_N2.P);
    d2Ap_TS3 = Calc2dAP(net,TS_N3.P);
    d2Ap_TS4 = Calc2dAP(net,TS_N4.P);
    d2Ap_TS5 = Calc2dAP(net,TS_N5.P);
    
    d2Ap_TS = [d2Ap_TS1 d2Ap_TS2 d2Ap_TS3 d2Ap_TS4 d2Ap_TS5];

    %Unnormalize the 2nd derivatives
    d2Ap_TR_UN = postmnmxDf2(d2Ap_TR,TRN.minp,TRN.maxp,TRN.mint,TRN.maxt,R,Qtr);
    d2Ap_TS_UN = postmnmxDf2(d2Ap_TS,TRN.minp,TRN.maxp,TRN.mint,TRN.maxt,R,Qts);
    
    %Unnormalize the input data (only for training, since testing is unnormalized)
    TRN_UN.P = postmnmx(TRN.P,TRN.minp,TRN.maxp);
    
    %Get the true 2nd derivatives of the sinc function
    [T_TR,Df_TR,Df2_TR] = H2Br_Dist(TRN_UN.P);
    [T_TS,Df_TS,Df2_TS] = H2Br_Dist(TS_UN{j}.P);
    
    %Compute and collect errors
    RMSE_TR_2nd = [RMSE_TR_2nd sqrt(mse(Df2_TR-d2Ap_TR_UN))];
    RMSE_TS_2nd = [RMSE_TS_2nd sqrt(mse(Df2_TS-d2Ap_TS_UN))];
    Err_TS_2nd = d2Ap_TS_UN - Df2_TS;
    
    save([save_dir '\Analysis\' 'Error_2nd_' num2str(j)],'Err_TS_2nd');
    
    clear Df2_TS d2Ap_TS_UN Df2_TR d2Ap_TR_UN T_TR T_TS Df_TR Df_TS d2Ap_TS Err_TS_2nd

end

save([save_dir 'Error_2nd' Ext_Load],'RMSE_TR_2nd','RMSE_TS_2nd');

