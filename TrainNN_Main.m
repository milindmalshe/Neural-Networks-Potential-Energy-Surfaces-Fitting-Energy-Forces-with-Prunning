
clear;
nntwarn off;
warning('off');

load_data = 'C:\Users\Admin\Documents\Google Talk Received Files\PhDCodes\PhDCodes\Data\2D Sinc\300 Points\';
load_dir = 'C:\Users\Admin\Documents\Google Talk Received Files\PhDCodes\PhDCodes\Data\2D Sinc\300 Points\';
save_dir = 'C:\Users\Admin\Documents\Google Talk Received Files\PhDCodes\PhDCodes\Data\2D Sinc\300 Points\Original\';
Ext_Load = '';
Ext_Save = '';

%GENERATE DATA
GenValidation = 0;
func_no = 4; 
M=10;
        
%TRAINING PARAMETERS
%DEFINE NO. OF NEURONS
neurons = 2*ceil([5 30 20 100 100 45 150]); 
ntrn = 2; %VALUES FOR RETRAINING
trnFcn = ['lm_df'];

%START LOADING DATA, TRAINING NN, SAVING NN AND ERRORS  
RMSE_TR = [];
RMSE_TR_trim = [];
RMSE_VD = []; RMSE_DfVd = [];
RMSE_TS = [];
RMSE_DfTr = [];
RMSE_DfTr_trim = [];
RMSE_DfTs = [];
Epoch = [];
d2AP_TRtrim = [];
d2AP_TS = [];
            
%LOAD VARIABLES
filename_L = ['Data_f' num2str(func_no) Ext_Load];
load([load_data filename_L]);
            
%TRAINING FOR M MONTE CARLO
for j=1:M,
    
    %ARRANGE DATA SET FOR TRAININ
    [TRN,vv] = ArrangeData(trnFcn,TR{j},VD{j},GenValidation);
    
    %CREATE NETWORKS                
    S1=neurons(func_no);
    S2=1;
    TF1='tansig';
    TF2='purelin';
            
    ntrain = 1;
    while ntrain < ntrn,
                    
        %SET PARAMETERS
        netn = newff(minmax(TRN.P),[S1 S2],{TF1 TF2});
        netn = initlay(netn);
        initnet = netn;
                  
        %filename_LL = [trnFcn num2str(j) '_f' num2str(func_no) phs Ext_Load];
        %load([load_dir filename_LL]);
        %netn = InitNet;
        %netn = net; tr_temp = tr;
        %initnet = InitNet;

        netn.trainFcn=['train' trnFcn];
        netn.trainParam.retrain=0;
        netn.trainParam.epochs=100;
        netn.trainParam.goal=0;%1e-16;
        netn.performFcn='mse';
        netn.trainParam.max_fail=Inf;
        netn.trainParam.show=10000;
        netn.trainParam.min_grad=0;%1e-12;
        Pi = []; Ai = [];                    
                    
        %TRAIN NN
        [netn,tr_temp]=train(netn,TRN.P,TRN.T,Pi,Ai,vv);
                    
        %COMPUTE NN OUTPUT AND ITS DERIVATIVE OF TRAINING DATA
        error = tr_temp.perf(length(tr_temp.perf));
            
        %RETRAIN TO SEE IF FALLING IN LOCAL MINIMUM
        if error < 1e-15,
            net = netn;
            tr = tr_temp;
            ntrain = ntrn;
            InitNet = initnet;
        else
            if ntrain == 1,
                net = netn;
                tr = tr_temp;
                error_old = error;
                InitNet = initnet;
            else
                error_new = error;
                if error_new < error_old,
                    net = netn;
                    tr = tr_temp;
                    error_old = error_new;
                    InitNet = initnet;
                end
            end
            ntrain = ntrain+1;
        end

        %TRIM DOWN THE TRAINING REGION (FOR 1D & 2D Problems ONLY)
        if isequal(func_no,1) || isequal(func_no,2) || isequal(func_no,3) || isequal(func_no,4),
            TRN_trim = trimTR(TRN,func_no);
        else
            TRN_trim = TRN;
        end

        %COMPUTE NN OUTPUT OF TRAINING & VALIDATION DATA
        %THIS IS THE BEST NN OUT OF THE ntrain NETWORKS.
        %train errors
        [Atr_trim,dAtr_trim,RMSEtr_trim] = NNResp(net,TRN_trim);
        [Atr,dAtr,RMSEtr] = NNResp(net,TRN);
        
        %test errors
        if isequal(func_no,5) || isequal(func_no,6) || isequal(func_no,7),
            [Ats,dAts,RMSEts] = NNResp(net,TS{j});
        else
            [Ats,dAts,RMSEts] = NNResp(net,TS);
        end

        %validation errors, if applicable
        if isempty(vv),
            RMSEvd.F = NaN; RMSEvd.Df = NaN;
        else
            if isequal(vv.DfvdFlag,0),
                RMSEvd.F = NaN; RMSEvd.Df = NaN;
            else
                [Avd,dAvd,RMSEvd] = NNResp(net,VD{j});
            end
        end
    
        %SHOW RESULTS
        fprintf('train%s',trnFcn);
        fprintf(', Function %g',func_no); 
        fprintf(', Data No. %g',j);
        fprintf(', tr %g',RMSEtr.F/10^-6);
        fprintf(', tr_trim %g',RMSEtr_trim.F/10^-6);
        fprintf(', vd %g',RMSEvd.F/10^-6);
        fprintf(', DfTr %g',RMSEtr.Df/10^-6);
        fprintf(', DfTr_trim %g',RMSEtr_trim.Df/10^-6);
        fprintf(', ts %g',RMSEts.F/10^-6);
        fprintf(', DfTs %g',RMSEts.Df/10^-6);
        fprintf(', Epoch %g',length(tr.epoch));
        %fprintf(', normgX %g',tr.normgX(length(tr.epoch))/1e-6);
        fprintf('\n');
        fprintf('Length %g/%g',length(TRN.P),sum(TRN.DfMap));
        fprintf('--------------------------------------');
        fprintf('\n');
                
        %SAVE Network
        filename_S = [trnFcn num2str(j) '_f' num2str(func_no) Ext_Save];
        save([save_dir filename_S],'net','tr','InitNet');
                            
        %SAVE ERROR FOR A SPECIFIC FUNCTION
        RMSE_TR = [RMSE_TR RMSEtr.F];
        RMSE_TR_trim = [RMSE_TR_trim RMSEtr_trim.F];
        RMSE_VD = [RMSE_VD RMSEvd.F];
        RMSE_TS = [RMSE_TS RMSEts.F];
        RMSE_DfTr = [RMSE_DfTr RMSEtr.Df];
        RMSE_DfTr_trim = [RMSE_DfTr_trim RMSEtr_trim.Df];
        RMSE_DfVd = [RMSE_DfVd RMSEvd.Df];
        RMSE_DfTs = [RMSE_DfTs RMSEts.Df];
        Epoch = [Epoch length(tr.epoch)];
                
        %Second Derivative Error (For 1D Problems ONLY)
        if isequal(func_no,1) || isequal(func_no,2) || isequal(func_no,3),
            D2Ap_TRtrim = Calc2dAP(net,TRN_trim.P);
            D2Ap_TS = Calc2dAP(net,TS.P);
            RMSE_d2AP_TRtrim = [d2AP_TRtrim sqrt(mse(TRN_trim.Df2-D2Ap_TRtrim))];
            RMSE_d2AP_TS = [d2AP_TS sqrt(mse(TS.Df2-D2Ap_TS))];
        else

            %For 2D Sinc, use Error_2nd.m for Unnormalized 2nd Derivative errors.
            
            %For H2Br, use Error_2nd_H2Br.m for Unnormalized 2nd Derivative errors.
            
            RMSE_d2AP_TRtrim = [];
            RMSE_d2AP_TS = [];
            
        end
                   
        %Save errors for each Monte Carlo
        save([save_dir 'temp'],'RMSE_TR','RMSE_TR_trim','RMSE_VD',...
            'RMSE_TS','RMSE_DfTr','RMSE_DfTr_trim','RMSE_DfVd','RMSE_DfTs',...
            'Epoch','RMSE_d2AP_TRtrim','RMSE_d2AP_TS');
    
    end
end

avg_tr_rmse = mean(RMSE_TR);
avg_trTrim_rmse = mean(RMSE_TR_trim);
avg_vd_rmse = mean(RMSE_VD);
avg_ts_rmse = mean(RMSE_TS);
avg_dfts_rmse = mean(RMSE_DfTs);
avg_dftr_rmse = mean(RMSE_DfTr);
avg_dftrTrim_rmse = mean(RMSE_DfTr_trim);
std_tr_rmse = std(RMSE_TR);
std_trTrim_rmse = std(RMSE_TR_trim);
std_vd_rmse = std(RMSE_VD);
std_ts_rmse = std(RMSE_TS);
std_dftr_rmse = std(RMSE_DfTr);
std_dftrTrim_rmse = std(RMSE_DfTr_trim);
std_dfts_rmse = std(RMSE_DfTs);
sstr = sort(RMSE_TR);
sstr_trim = sort(RMSE_TR_trim);
ssvd = sort(RMSE_VD);
ssts = sort(RMSE_TS);
ssdftr = sort(RMSE_DfTr);
ssdftr_trim = sort(RMSE_DfTr_trim);
ssdfts = sort(RMSE_DfTs);
mnE_tr = sstr(1); mdE_tr = median(sstr); mxE_tr = sstr(ceil(0.6*M));
mnE_tr_trim = sstr_trim(1); mdE_tr_trim = median(sstr_trim); mxE_tr_trim = sstr_trim(ceil(0.6*M)); 
mnE_vd = ssvd(1); mdE_vd = median(ssvd); mxE_vd = ssvd(ceil(0.6*M));
mnE_ts = ssts(1); mdE_ts = median(ssts); mxE_ts = ssts(ceil(0.6*M));
mnE_dftr = ssdftr(1); mdE_dftr = median(ssdftr); mxE_dftr = ssdftr(ceil(0.6*M));
mnE_dftr_trim = ssdftr_trim(1); mdE_dftr_trim = median(ssdftr_trim); mxE_dftr_trim = ssdftr_trim(ceil(0.6*M));
mnE_dfts = ssdfts(1); mdE_dfts = median(ssdfts); mxE_dfts = ssdfts(ceil(0.6*M));
        
%SHOW AVERAGE
fprintf('avg_tr %g',avg_tr_rmse/10^-6);
fprintf(', avg_vd %g',avg_vd_rmse/10^-6);
fprintf(', avg_ts %g',avg_ts_rmse/10^-6);
fprintf(', avg_df %g',avg_dftr_rmse/10^-6);
fprintf(', mdE_tr %g',mdE_tr/10^-6);
fprintf(', mdE_tr_trim %g',mdE_tr_trim/10^-6);
fprintf(', mdE_vd %g',mdE_vd/10^-6);
fprintf(', mdE_dftr %g',mdE_dftr/10^-6);
fprintf(', mdE_dftr_trim %g',mdE_dftr_trim/10^-6);
fprintf(', mdE_ts %g',mdE_ts/10^-6);
fprintf(', mdE_dfts %g',mdE_dfts/10^-6);
fprintf('\n');
fprintf('====================================================');
fprintf('\n');
fprintf('====================================================');
fprintf('\n');
    
filename_err = ['Error_f' num2str(func_no) '_' trnFcn Ext_Save];
save([save_dir filename_err],'RMSE_TR','RMSE_TR_trim','RMSE_VD','RMSE_TS','RMSE_DfTr','RMSE_DfTr_trim','RMSE_DfVd','RMSE_DfTs',...
    'avg_tr_rmse','avg_trTrim_rmse','avg_vd_rmse','avg_ts_rmse','avg_dftr_rmse','avg_dftrTrim_rmse','avg_dfts_rmse',...
    'std_tr_rmse','std_trTrim_rmse','std_vd_rmse','std_ts_rmse','std_dftr_rmse','std_dftrTrim_rmse','std_dfts_rmse',...
    'mnE_tr','mxE_tr','mnE_vd','mxE_vd','mnE_ts','mxE_ts',...
    'mnE_tr_trim','mxE_tr_trim','mdE_tr_trim',...
    'mnE_dftr','mxE_dftr','mdE_dftr',...
    'mnE_dftr_trim','mxE_dftr_trim','mdE_dftr_trim',...
    'mnE_dfts','mxE_dfts','mdE_tr','mdE_vd','mdE_ts','mdE_dfts',...
    'RMSE_d2AP_TRtrim','RMSE_d2AP_TS');
