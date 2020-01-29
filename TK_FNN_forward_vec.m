
function Y = TK_FNN_forward_vec(XI,W,b,h_modes)
% Calls FNN_forward given serveal input vectors. See FNN_forward for 
% additional description. 
% 
% Parameters
% ----------
% XI : N x D matrix
%     N input vectors to the FNN network where n is the amout of inputs. 
% See FNN_forward for additional parameters.
% 
% Returns
% -------
% Y : N x D_M column vector
%     FNN output for each input where D_M is the dimension of the last 
%     layer. 

%% Initialise variables
[N ,  ~ ] = size(XI);
[D_M, ~] = size(W{end}); % Output dimension
Y = nan(N,D_M); 

%% Evaluate network in loop 
for i = 1:N
    % Estimate and save as row vector
    Y(i,:) = TK_FNN_forward(XI(i,:),W,b,h_modes);
end 
end