
function [g_c, g_sigma,g_theta] = TK_NF_grad(x,d,c,sigma,theta)
% Derivative with respect to 0.5 times the squared error objective 
% function for the Takagi-Sugeno artifical neuro-fuzzy network in
% TK_NF_forward-
% 
% References
% ----------
% Compitational Intelligence in Time Series Forecasting by Palit, Ajoy K 
% and Popovic, Dobrivoje,  2015 section 6.4.2
% 
% Parameters
% ----------
% x : 1 x n row vector
%     Input vector to the TS network where n is the amout of inputs. 
% d : float
%     Target output
% See TK_NF_forward for additional parameters. 
% 
% Returns
% -------
% g_c : M x n matrix
%     The gradient (partial S)/(partial c), where S = 0.5*(d - TS(x))^2
%     and TS is the forward propagation of the TS-ANFIS. 
% g_sigma : M x n matrix
%     Simmilar to g_c but gradient with respect to sigma parameter. 
% g_theta : M x (n + 1) matrix
%     Simmilar to p_theta but gradient with respect to theta parameter. Note
%     that g_theta (l,1) is the gradient with respect to the bias theta(l,1).

%% Sanity checks
if numel(d) ~= 1
    error('NF network only supports single output. Make sure target output is not vector.');
end 


%% Initally run forward propagation 
[f, y_TS, gamma] = TK_NF_forward(x,c,sigma,theta); 
err = f - d; 

%% Compute gradient with respect to c
A = (y_TS - f)*err; % Temporary constant needed
g_c = 2.*A.*gamma.*(x - c)./(sigma.^2); 

%% Compute gradient with respect to sigma
g_sigma = 2.*A.*gamma.*((x - c).^2)./(sigma.^3);

%% Finally compute gradient with respect to theta
x_ext = [1 x]; 
% gamma*x is equivilent to the outer product between the two had x_ext 
% been a column vector. 
g_theta = err.*(gamma*x_ext); 
end 
