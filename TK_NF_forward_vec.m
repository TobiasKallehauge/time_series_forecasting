 
function Y = TK_NF_forward_vec(XI,c,sigma,theta)
% Calls TK_NF_forward given serveal input vectors. See TK_NF_forward for
% additional description. 
% 
% Parameters
% ----------
% XI : N x n matrix
%     N input vectors to the NF network where n is the amout of inputs. 
% See TK_NF_forward for additional parameters.
% 
% Returns
% -------
% Y : N x 1 column vector
%     NF output for each input. 


%% Initialise variables
[N,  ~ ] = size(XI);
Y = nan(N,1); 

%% Evaluate network in loop 
for i = 1:N
    Y(i) = TK_NF_forward(XI(i,:),c,sigma,theta); % Estimate
end 
end