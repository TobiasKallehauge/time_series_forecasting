

function para = TK_NF_init_para(data,D,M)
% Initialise the parameters for a neuro fuzzy network based on the dataset
% and number of rules. See TK_NF_forward for further description of the
% network.
% 
% Parameters
% ----------
% data : column vector
%   Training data for the NF network. The center parameters c in the neuru
%   fuzzy network will be initialised from uniform distribution on the
%   range of the data.
% D : positive int
%   Number of inputs to the NF network.     
% M : positive int
%   Number of rules in the NF network.
%
% Returns
% -------
% para : cell
%   Cell containing matricies with the parameters where: 
%   para{1} = c: GMF mean parameters
%   para{2} = sigma: GMF starndard deviation parameters
%   para{3} = theta: TS rule parameters


%Initialize c0 in the range of the time series
dat_min = min(data); dat_max = max(data);
c = (dat_max - dat_min)*rand(M,D) + dat_min; 
sigma = rand(M,D);
theta = rand(M,D + 1); 
para = {c,sigma,theta}; 
end