
function [g_W, g_b] = TK_FNN_grad(x,d,W,b,h_modes)
% Derivative with respect to 0.5 times the squared error objective function 
% for the FNN computed by backpropagation. The gradient is derived with
% respect to just a single training example. 
% 
% Parameters
% ----------
% x : 1 x D row vector
%     Input vector to the FNN network where D is the amout of inputs. 
% d : 1 x D_M
%     Target output where D_M is the output dimension of the network, 
% See FNN_forward for additional parameters. 
% 
% Returns
% -------
% g_W : 1 x M cell 
%     The gradient of the weights W{l} for each layer l. 
%     The gradient is computed as (partial S)/(partial W{l}), 
%     where S = 0.5*(d - FNN(x))^2 and FNN is the forward propagation of
%     the FNN_forward. 
% g_b : 1 x M cell
%     Simmilar to g_W but gradient with respect to biases. 


%% Initally run forward propagation and setup relevant variables
[y, a_all, z_all] = TK_FNN_forward(x,W,b,h_modes); % Forward propagation
[~, M] = size(W); % Number of layers
g_W = cell(1,M); % Gradient of weights
g_b = cell(1,M); % Gradient of biasses

% Get derivative of activation functions.
d_h = cell(1,M);
for l = 1:M
    switch h_modes(l)
        case 0
            d_h{l} = @d_identity; % The @ handles the function as a variable
        case 1
            d_h{l} = @d_tanh;
        case 2
            d_h{l} = @d_sigmoid;
        case 3
            d_h{l} = @d_ReLU; 
    end 
end 

%% Run backpropagation
delta = ((y - d)').*d_h{M}(a_all{M}); % Last layer has different delta
g_W{M} = delta*(z_all{M}');
g_b{M} = delta; 
for l = (M-1):-1:1 % Second last layer to the first layer
    % Update error term
    % Weight W{l+1} is needed due to how indexing is done. 
    delta = d_h{M}(a_all{l}).*((W{l+1}')*delta); 
    
    % Compute gradients 
    g_W{l} = delta*(z_all{l}');
    g_b{l} = delta; 
end 
end 

%% Various derivatives of activation functions 
function y = d_identity(a)
%{
Derivative of identity activation function of input a.
%}
y = ones(length(a),1);
end 

function y = d_tanh(a)
%{
Derivative of hyberboliic tanget activation function of input a.
%}
y = 1 - tanh(a).^2; 
end

function y = d_sigmoid(a)
%{
Derivative of Sigmoid activation function of input a.
%}
tmp = exp(-a); % Save some computation time
y = tmp./((1 + tmp).^2);
end

function y = d_ReLU(a)
%{
Derivative of Recitfier liniear unit activation function of input a.
%}
y = a > 0; 
end
