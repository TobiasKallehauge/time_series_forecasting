
function [y, y_TS, gamma] = TK_NF_forward(x,c,sigma,theta)
% Implementation of MISO (multible input - single output) forward propagation 
% of Takagi-Sugeno artifical neuro-fuzzy network (TS-ANFIS or NF). Gaussian 
% membership functions are utilized for the degree of membership assignment
% . Multible outputs are not supported. 
% 
% References
% ----------
% Compitational Intelligence in Time Series Forecasting by Palit, Ajoy K 
% and Popovic, Dobrivoje, 2015 section 6.4 and 6.5
% 
% Parameters
% ----------
% x : 1 x n row vector
%     Input vector to the NF network where n is the amout of inputs. 
% c : M x n matrix
%     Mean variable where c(l,i) is the mean for the i'th input fuzzy set
%     for the l'th rule with gaussian membership function.
%     M is the amout of rules. 
% sigma : M x n matrix
%     Standard deviation variable where sigma(l,i) corrosponds to the
%     variance of the i'th input fuzzy set for the l'th rule with gaussian
%     membership function. Elements are allowed to be negative but are 
%     squared in the function. 
% theta : M x (n + 1) matrix
%     Coefficients for NF rules where theta(l,i) is multiplied by x_i in 
%     the l'th rule. theta(l,1) is the bias for each rule.
% 
% Returns
% -------
% y : float
%     Single output for the network. 

%% Compute degree of fullfullment of each rule
% x and c dimensions does not match but subtraction does compute correctly.
tmp = ((x - c)./sigma).^2; 
% Sum over the columns (2. dimension) and take elementwise exponential. 
beta = exp(-sum(tmp,2)); %degree of fulfillment 
gamma = beta./sum(beta); %normalised degree of fulfillment

%% Compute linear output of each NF rule
% Introduce x_1 = 1 in extended input transform input to column vector
x_ext = ([1 x])'; 
y_TS = theta*x_ext; % Diffent linear combinations of input for each rule

%% Finally compute network output
y = dot(y_TS,gamma); 
end 
