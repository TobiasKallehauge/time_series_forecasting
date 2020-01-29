clear; close all;  

%% Inport data and split into training/validation/test data
D = 4; % number of inputs
d = 6; % seasonal constant
L = 6; % prediction leap

% Load file (change based on which timeseries is wanted)
file_path = 'data/mgdata.mat'; % <-- change here use other files
load(file_path);
data = mgdata(:,2); % Only 2. row is needed 


% Do preprocessing (change "mgdata" to whatever file is loaded)
[XI, XO] = TK_timeseries_preprocess(data,D,d,L); % preprocess data
XI_trn = XI(101:300,:); XO_trn = XO(101:300); % training data
XI_tst = XI(301:end,:); XO_tst = XO(301:end); % test data

%% Initialise TS_ANFIS
M  = 8; % number of rules
para0 = TK_NF_init_para(data,D,M);

%% Setup SQD
% Required arguments
eta = 1e-2; % learning rate
epochs = 1000; % Maximum number of training epochs
mode = 1; % 0: FNN, 1: NF

% Important optional arguments
verbose = 1; % 1 - display progress, 0 - display nothing
early_stop_epoch = 0; % 0: No early stopping, 1: Do early stopping
N_early_stop = 100; % Stop when SSE have not been lowered for so many epochs
early_stop_tol = 0; % 0: No early stopping, 1: Do early stopping
tol = 5e-2; % Stop when SSE is lower than tol
batch_size = 1; % How many training examples gradients is updated with respect to

%% Run SQD
[para,obj,hist] = TK_optimize(XI_trn,XO_trn,para0,mode,eta,epochs,...
    'verbose',verbose,'batch_size',batch_size,...
    'early_stop_tol',early_stop_tol, 'tol', tol, ....
    'early_stop_epoch',early_stop_epoch, 'N_early_stop',N_early_stop);
[c, sigma, theta] = para{:}; 

% Plot training history
figure;
loglog(hist);
grid on; 
title('Training history');
xlabel('Epoch'); ylabel('SSE');
fprintf('Training MSE = %.2e \n',obj/length(XO_trn));

%% Evaluate on test data
XO_tst_est = TK_NF_forward_vec(XI_tst,c,sigma,theta);

% Use precalculated parameters
load('data/TS_para.mat');
[c_precal, sigma_preacal, theta_precal] = para{:}; 
XO_tst_precal = TK_NF_forward_vec(XI_tst,c_precal,sigma_preacal,...
    theta_precal);

% Plot
figure; 
plot(XO_tst); hold on; 
plot(XO_tst_est); hold on; 
plot(XO_tst_precal); 
legend('Target','Estimated','Estimated - precal para');

% Print out some statistics 
MSE_val = sum((XO_tst_est - XO_tst).^2)/length(XO_tst_est);
RMSE_val = sqrt(sum((XO_tst_est - XO_tst).^2)/length(XO_tst_est));
MAE_val = sum(abs((XO_tst_est - XO_tst)))/length(XO_tst_est);
MSE_val_precal = sum((XO_tst_precal - XO_tst).^2)/length(XO_tst_est);
RMSE_val_precal = sqrt(sum((XO_tst_precal - XO_tst).^2)/length(XO_tst_est));
MAE_val_precal = sum(abs((XO_tst_precal - XO_tst)))/length(XO_tst_est);
fprintf('Test MSE = %.2e, RMSE = %.2e, MAE = %.2e\n',...
    MSE_val, RMSE_val,MAE_val);
fprintf('Test MSE = %.2e, RMSE = %.2e, MAE = %.2e (precalculated parameters)\n',...
    MSE_val_precal,RMSE_val_precal,MAE_val_precal);
