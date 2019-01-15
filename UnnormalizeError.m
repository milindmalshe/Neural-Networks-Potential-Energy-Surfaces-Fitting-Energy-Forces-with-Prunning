clear
nntwarn off

trnFcn = 'bfg_df'; 
func_no=6;
M = 1; %For 2D sinc and H2Br, change M to 10.
   
% load_data = 'C:\Documents and Settings\Administrator\Desktop\PhD Codes\Data\Si5\1500 Points\';
% load_dir = 'D:\Ohm\Neuron Removal\Si5\1500TR\xDf_I\Df_closept\Thrd_mxmndist\0.1Ratio\Trainbfg_df\From 50Kepochs\';

load_data = 'C:\Users\Admin\Documents\Google Talk Received Files\PhDCodes\PhDCodes\Data\Si5\1500 Points\';
load_dir = 'D:\Ohm\Neuron Removal\Si5\1500TR\xDf_I\Df_closept\Thrd_mxmndist\0.1Ratio\Trainbfg_df\From 50Kepochs\';



load([load_data 'Data_f' num2str(func_no)]);

UN_RMSEtr.F = []; UN_RMSEtr.Df = []; UN_RMSEts.F=[]; UN_RMSEts.Df=[];
N_RMSEts.F = []; N_RMSEts.Df = []; S1=[];

for i=1:M,

    %Obtain Data
    load([load_dir trnFcn num2str(i) '_f' num2str(func_no)]);
    TRN = TR{i};
    TS = TS{i};

    %Verify Error
    [Atr,dAtr,RMSEtr] = NNResp(net,TRN);
    [Ats,dAts,RMSEts] = NNResp(net,TS);

    N_RMSEts.F = [N_RMSEts.F RMSEts.F];
    N_RMSEts.Df = [N_RMSEts.Df RMSEts.Df];

    %Un-Normalization
    %--> TRAIN
    TRN_UN{i}.T = postmnmx(TRN.T,TRN.mint,TRN.maxt);
    TRN_UN{i}.Df = postmnmxDf(TRN.Df,TRN.minp,TRN.maxp,TRN.mint,TRN.maxt);

    Atr_UN{i} = postmnmx(Atr{2},TRN.mint,TRN.maxt);
    dAtr_UN{i} = postmnmxDf(dAtr{2},TRN.minp,TRN.maxp,TRN.mint,TRN.maxt);

    %--> TEST
    TS_UN{i}.T = postmnmx(TS.T,TRN.mint,TRN.maxt);
    TS_UN{i}.Df = postmnmxDf(TS.Df,TRN.minp,TRN.maxp,TRN.mint,TRN.maxt);

    Ats_UN{i} = postmnmx(Ats{2},TRN.mint,TRN.maxt);
    dAts_UN{i} = postmnmxDf(dAts{2},TRN.minp,TRN.maxp,TRN.mint,TRN.maxt);

    %Compute Un-Normalized Error
    %-->TRAIN
    TR_UN_Error{i}.F = abs(TRN_UN{i}.T-Atr_UN{i});
    TR_UN_Error{i}.Df = reshape(abs(TRN_UN{i}.Df-dAtr_UN{i}),length(TRN_UN{i}.Df)/net.inputs{1}.size,net.inputs{1}.size)';
    TR_RMSE_UN.F = sqrt(mse(TR_UN_Error{i}.F));
    TR_RMSE_UN.Df = sqrt(mse(TR_UN_Error{i}.Df));

    %-->TEST
    TS_UN_Error{i}.F = abs(TS_UN{i}.T-Ats_UN{i});
    TS_UN_Error{i}.Df = reshape(abs(TS_UN{i}.Df-dAts_UN{i}),length(TS_UN{i}.Df)/net.inputs{1}.size,net.inputs{1}.size)';

    TS_RMSE_UN.F = sqrt(mse(TS_UN_Error{i}.F));
    TS_RMSE_UN.Df = sqrt(mse(TS_UN_Error{i}.Df));

    UN_RMSEtr.F = [UN_RMSEtr.F TR_RMSE_UN.F];
    UN_RMSEtr.Df = [UN_RMSEtr.Df TR_RMSE_UN.Df];
    UN_RMSEts.F = [UN_RMSEts.F TS_RMSE_UN.F];
    UN_RMSEts.Df = [UN_RMSEts.Df TS_RMSE_UN.Df];

    TS_UN{i} = [];

    fprintf('train%s',trnFcn);
    fprintf(', Function %g',func_no); 
    fprintf(', Data No. %g',i);
    fprintf(', tr %g',TR_RMSE_UN.F/10^-6); 
    fprintf(', tr_trim %g',TR_RMSE_UN.F/10^-6);
    fprintf(', DfTr %g',TR_RMSE_UN.Df/10^-6);
    fprintf(', DfTr_trim %g',TR_RMSE_UN.Df/10^-6);
    fprintf(', ts %g',TS_RMSE_UN.F/10^-6);
    fprintf(', DfTs %g',TS_RMSE_UN.Df/10^-6);
    fprintf(', Epoch %g',length(tr.epoch));
    fprintf('\n');
    fprintf('Length %g/%g',length(TRN.P),sum(TRN.DfMap));
    fprintf('--------------------------------------');
    fprintf('\n');    

    %NEURON
    S1(i) = length(net.IW{1,1});

end

fprintf('avg_tr %g',mean(UN_RMSEtr.F)/10^-6);
fprintf(', avg_ts %g',mean(UN_RMSEts.F)/10^-6);
fprintf(', avg_df %g',mean(UN_RMSEtr.Df)/10^-6);
fprintf(', mdE_tr %g',median(UN_RMSEtr.F)/10^-6);
fprintf(', mdE_tr_trim %g',median(UN_RMSEtr.F)/10^-6);
fprintf(', mdE_dftr %g',median(UN_RMSEtr.Df)/10^-6);
fprintf(', mdE_dftr_trim %g',median(UN_RMSEtr.Df)/10^-6);
fprintf(', mdE_ts %g',median(UN_RMSEts.F)/10^-6);
fprintf(', mdE_dfts %g',median(UN_RMSEts.Df)/10^-6);
fprintf('\n');
fprintf('====================================================');
fprintf('\n');
fprintf('====================================================');
fprintf('\n');

filename_err = 'UN_Error';
save([load_dir filename_err],'UN_RMSEtr','UN_RMSEts','S1');