
function para = TK_FNN_init_para(M,D,h_modes)
% Initialise parameters for feed forward neural network. Weights and
% are initialised on a uniform distribution between 0 and 1, the biasses 
% are initialised as zero and the activation functions  are included in the 
% return parameters vector. See TK_FNN_forward for further descrtions of 
% the parameters.
% 
% Returns
% -------
% para : cell
%   Cell containing matricies with the parameters where: 
%   para{1} = W: Weights.
%   para{2} = b: biasses.
%   para{3} = h: Activation functions. 

D_FNN = [repmat(D,1,M) 1]; 
W0 = cell(1,M); b0 = cell(1,M);
for l = 1:M
    W0{l} = rand(D_FNN(l+1),D_FNN(l)); % Weights are initialised randomly
    b0{l} = zeros(D_FNN(l+1),1); % Biasses are set to zero
end
para = {W0,b0,h_modes};
end 
