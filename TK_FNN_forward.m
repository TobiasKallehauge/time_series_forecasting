
function [y, a_all, z_all] = TK_FNN_forward(x,W,b,h_modes)
% Implementation of standard Feed Forward Neural Network (forward 
% propagation). Dimensions of network are set by the given weights and
% biasses, and activation functions are chosen from a set of available ones
% (see parameter description). 
% 
% References
% ----------
% Pattern Recognition and Machine learning by Bishop,Christopher M. page 
% 227-246
% Compitational Intelligence in Time Series Forecasting by Palit, Ajoy K 
% and Popovic, Dobrivoje, 2015 section 3.3.1 page 84-85
% 
% 
% Parameters
% ----------
% x : 1 x D row vector
%     Input vector to the FNN network where D is the amout of inputs. 
% W : 1 x M cell  
%     M weights of the network where W{l} are the weights for layer l where
%     M is the number of layers. If W{l} has dimension (D_l x D_(l-1)) then
%     D_l is the number of nodes in layer l. The sizes of weight
%     matrices are expected to be consistent with  eachother:
%     i.e. #columns(W{l-1}) = #rows(W{l}).
% b : 1 x M cell
%     Biasses of the network where b{l} are the biasses for layer l
%     simmilar to W. 
% h_modes : M x 1 vector
%     Vector where h(l) is the choise of activation function for layer l
%     with 4 possible choises:
%        h(l) = 0 : Identity : a
%        h(l) = 1 : Hyperbolic tangent: (exp(a) - exp(-a))/(exp(a)+exp(-a))
%        h(l) = 2 : Sigmoid : 1/(1+exp(-a))
%        h(l) = 3 : Relu : max(0,a)        
%     h(l) can be choosen differently for each layer. For regression
%     problems h(M) = 0 is recommended. 
% 
% Returns
% -------
% y : D_M x 1 vector
%     Output for the network where D_M = #columns(W{M})
% a_all : 1 x M cell
%     Cell where each element a_all{l} contains the acivations (values 
%     after linear operations) for each layer l (needed in backpropagation)
% z_all : 1 x M cell
%     Simmilar to a_all but for values in the nodes (values after
%     non-linear operation). 

%% Initialise network
[~ , M] = size(W); % Number of layers
% Get choises of activativation functions as cell with functions handles
h = cell(1,M); % Cell with various activation functiuons 
for l = 1:M
    switch h_modes(l)
        case 0
            h{l} = @identity; % The @ handles the function as a variable
        case 1
            h{l} = @tanh;
        case 2
            h{l} = @sigmoid;
        case 3
            h{l} = @ReLU; 
    end 
end 
% For backpropagation the values of nodes and activations in each layer
% is required. Create empty cells for these to be stored. 
a_all = cell(M); % activations
z_all = cell(M+1); % nodes


%% Perform forward propagation
z = x'; % First input (transform to column vector) 
z_all{1} = z;  % First nodes are considered the input
for l = 1:M
    % Compute activation by linear operation
    a = W{l}*z + b{l}; 
    
    % Evaluate each activation throug non-linear activation function. 
    z = h{l}(a);
    
    % Save z,a for gradient computation
    a_all{l} = a; 
    z_all{l+1} = z; 
end 
y = z'; % Network output from last layer (row vector). 
end 

function y = identity(a)
%{
Identity activation function of input a (needed in implementation).
%}
y = a;
end 

function y = sigmoid(a)
%{
Sigmoid activation function of input a.
%}

y = 1./(1 + exp(-a));
end

function y = ReLU(a)
%{
Recitfier liniear unit activation function of input a.
%}
y = max(a,0); 
end