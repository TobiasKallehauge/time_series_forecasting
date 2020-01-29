function [para,obj,hist] = TK_optimize(x,d,para0,mode,eta,max_epochs,...
    varargin) 
% Gradient descent type algorithm for for either TS-ANFIS or FNN using the 
% squared error as objetive. The gradient is updated with respect to one or
% more training examples which is specified by the batch size. Note that
% not all parameters are required (see parameter description). See
% TK_FNN_forward/TK_NF_forward a detailed description of the FNN/TS-ANFIS
% respectively including description about parametrisation. 
%
% References
% ----------
% Pattern Recognition and Machine learning by Bishop,Christopher M. page 
% 240-421
%
% Parameters (required)
% ---------------------
% x : N x n matrix
%     N input vectors to the TS network where n is the amout of inputs. 
% d : N x 1 column vector
%     N output target value for the TS network. 
% para0 : cell
%     Cell containing inital parameters for either TS-ANFIS of FNN. For FNN
%     it also contains hyperparameters h_modes.
% eta : positive float
%     Learning rate for the algorithm.
% mode : positive int
%     Which type of network is beeing trained
%         0 : FNN
%         1 : TS-ANFIS
% max_epochs : positive int
%     How many epochs to run the training data for before stopping. Other
%     stopping parameters are available in the optional parameters. If more
%     than one stoppping parameter is active the training algorithm stops
%     when either stopping criteria becomes true. 
%
% Parameters (optional)
% ---------------------
% batch_size : nonzero postive int
%     How many training examples the gradients should be updated with
%     respect to. For batch_size = 1 sequential/stochastic GD is performed
%     end for batch_size = #Training examples regular GD is performed.
%     Defaults to 1. 
% shuffle : bool
%     If true training data is shuffled between each epoch. If false data
%     is looped sequentially. Defaults to false. 
% verbose : bool
%     If true progress is displayed - else nothing is.
% displaysec : positive float
%     How many seconds between printing when verbose is true.
%     Defaults 0.5 seconds. 
% early_stoping_epoch : bool
%     If true stop training when SEE have not been lowered for N_early_stop
%     epochs. Defaults to false.
% N_early_stop : postive int
%     After how many epochs to do early stopping. Defaults to 100. Note
%     that this parameters is only checked for when early_stoping_epoch is
%     true. 
% early_stoping_tol : bool
%     If true stop training when SEE is lower than tol. Defaults to false.
% N_early_stop : postive non-zero float
%     Tolerence for SSE. Note that this parameters is only checked for when 
%     early_stoping_tol is true. 
% 
% Returns
% -------
% para : cell
%     Cell containing estimated paramers. 
% obj : float
%     Total error on training set with the estimated errors.
% hist : epochs x 1 vector
%     History of total error on training set after each epoch. 

%% Parse optional arguments
p = inputParser;
addParameter(p,'batch_size',1); 
addParameter(p,'shuffle',false);
addParameter(p,'verbose',true);
addParameter(p,'early_stop_epoch',false);
addParameter(p,'N_early_stop',100);
addParameter(p,'early_stop_tol',false);
addParameter(p,'tol',1e-3);
addParameter(p,'displaysec',0.1);
parse(p,varargin{:});

% Extract optional arguments as variables if used often
batch_size = p.Results.batch_size;

%% Perform some sanity cheks
[N,  ~] = size(x); % Number of training examples 
if batch_size > N
    error('Batch size = %d should be lower than the total amount of training examples = %d',batch_size,N);
end

if ~ ismember(mode,[0,1])
    error('mode should be either 0 for FNN or 1 for TS-ANFIS');
end 


%% Initialise optimization
hist = nan(1,max_epochs + 1); 
obj_i = SSE(x,d,para0,mode); %Inital objective value
hist(1) = obj_i; 
para_i  = para0; % Parameters that are updated in the loop
obj = obj_i; % Lower training error (updated in loop)
para = para_i; % The parameters with lowest training error (updated in loop)
progress_count = 0; % Count for how many epochs the training error does not fall

% Compute how many batches the training data is split into
N_batches = floor(N/batch_size);
% The last batch might have to be bigger
last_batch_size = batch_size + N - batch_size*N_batches;
batch_sizes = [repmat(batch_size,1,N_batches - 1), last_batch_size];

lineLength = 0; % Used to print progress
time0 = clock; time0 = time0(6); 
%% Run SQD for a specified number of epochs
for i = 1:max_epochs
    
    % Shuffle training data if prompted - stochastic gradient descent
    if p.Results.shuffle 
        perm = randperm(N);
    else
        perm = 1:N;
    end 
    
    x_perm = x(perm,:);
    d_perm = d(perm,:); 
    
    
    %% Run epoch
    for j = 0:(N_batches - 1) % loop over each minibatch
    
    %% Unpack parameters and initialise gradients
    % Initialise gradients with the first training example in minibatch
    x1 = x(j*batch_size+1,:);
    d1 = d(j*batch_size+1,:); 
    switch mode
        case 0 % FNN
            [W,b,h_modes] = para_i{:}; 
            % Get correct size gradients by 1 call of backpropagation 
            [g_W, g_b] = TK_FNN_grad(x1,d1,W,b,h_modes);
            M = length(g_W); % Number of layers 
        case 1 % TS-ANFIS
            [c,sigma,theta] = para_i{:}; 
            % Get correct size gradients by 1 call of backpropagation 
            [g_c, g_sigma,g_theta] = TK_NF_grad(x1,d1,c,sigma,theta);
    end 
    
    %% Run minibatches
    for k = 2:batch_sizes(j+1) % loop over each training example in minibatch
        idx = j*batch_size + k; % Current training example
        xk = x_perm(idx,:);
        dK = d_perm(idx,:);
        
        switch mode % Get sub gradients for training example
            case 0 % FNN        
                [g_W_k,g_b_k] = TK_FNN_grad(xk,dK,W,b,h_modes);
                
                % add subgradiens to total gradient
                for l = 1:M
                    g_W{l} = g_W{l} + g_W_k{l};
                    g_b{l} = g_b{l} + g_b_k{l};
                end 
                
            case 1 % TS-ANFIS
                
                [g_c_k, g_sigma_k,g_theta_k] = TK_NF_grad(...
                    xk,dK,c,sigma,theta); 
                
                % add subgradiens to total gradient
                g_c = g_c + g_c_k;
                g_sigma = g_sigma + g_sigma_k;
                g_theta = g_theta + g_theta_k;
        end 
    end 
    
    switch mode
        case 0 % FNN
            for l = 1:M
                W{l} = W{l} - eta*g_W{l}./batch_size;
                b{l} = b{l} - eta*g_b{l}./batch_size; 
            end 
        case 1 % TS-ANFIS
            % Update via SQD
            c = c - eta*g_c;
            sigma = sigma - eta*g_sigma./batch_size;
            theta = theta - eta*g_theta./batch_size; 

    end 
    
    % Finally pack parameters again
    switch mode
        case 0 % FNN 
            para_i = {W,b,h_modes}; 
        case 1 % TS_ANFIS 
            para_i = {c,sigma,theta};   
    end     
    end 
    
    
    %% Record objective value
    obj_i = SSE(x,d,para_i,mode); 
    hist(i + 1) = obj_i; 
    
    % Update the final output parameter if lower SSE is achieved
    if obj_i < obj
        obj = obj_i;
        para = para_i;
        progress_count = 0; 
    elseif p.Results.early_stop_epoch
        progress_count = progress_count + 1; 
        if progress_count == p.Results.N_early_stop
            fprintf('Training error has not been lowered for %d epochs. Stopping training!\n',progress_count);
            break; 
        end
    end 
    if p.Results.early_stop_tol && (obj <= p.Results.tol)
        fprintf('Desired tolerence met with SSE = %.2e. Stopping training!\n',obj);
        break;
    end
    
    if p.Results.verbose
        time1 = clock; time1 = time1(6); % Get seconds 
        if time1 - time0 > p.Results.displaysec || i == max_epochs
            time0 = time1; 
            
            % Set up progressbar
            progress_percent = i/max_epochs;
            tmp = round(40*progress_percent);
            if i < max_epochs
                arrow = '>';
            else
                arrow = '=';
            end 
            progressbar = sprintf('|%s%s%s|',repmat('=',1,tmp-1),...
                arrow,repmat(' ',1,40-tmp));
            
            fprintf(repmat('\b',1,lineLength)); % Clear line
            % Print all
            lineLength = fprintf('Progress: %.0f%% %s Epoch: %d, SSE = %.2e',...
                progress_percent*100,progressbar,i,obj); 
        end 
    end 
end 
if p.Results.verbose
    fprintf('\n');  
end 
end 

function obj = SSE(x,d,para,mode)
%{
Compute sum og squared errors (SSE) of TS-ANFIS network given input and 
target values

Parameters
----------
See TS_SGD for parameters. 

Returns
-------
obj : float
    Sum of squared errors (value of objective functuion). 
%}

switch mode
    case 0 % FNN
        [W,b,h_modes] = para{:}; 
        d_est = TK_FNN_forward_vec(x,W,b,h_modes); 
    case 1 % TS-ANFIS
        [c,sigma,theta] = para{:}; % Unpack parameters
        d_est = TK_NF_forward_vec(x,c,sigma,theta);
end 
obj = sum((d - d_est).^2); % SSE
end 
