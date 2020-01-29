 

function [XI,XO] = TK_timeseries_preprocess(data,D,d,L)
%{
Preprocess any timeseries data into input matrix XI and output matrix XO,
with the goal of predicting XI based on XO.

References
----------
Compitational Intelligence in Time Series Forecasting by Palit, Ajoy K 
and Popovic, Dobrivoje, 2015 section 4.6

Parameters
----------
data : column vector
    Data used for preprocessing. 
D : positive int
    Number of inputs network - i.e number of previous values including 
    t used to predict.
d : postive int
    Seasonal constant - i.e temporal spacing between input values. Input 
    becomes:
    XI(t) = [X(t - (D-1)d), X(t - (D-2)d), ... , X(t)]
    where X is the MG time series and t is the descrete time. 
L : postive int
    Prediction step - i.e how far in time is predicted. Output becomes:
    XO(t) = X(t + L)

Returns
-------
XI : y x D matrix
    Input matrix for the specified parameters. Size depends on choise of 
    D, d and L.
XO : y x 1 vector
    Output matrix for the specified paramters. Size depends on choise of 
    D, d and L.
%}

%% Sanity checks on the input
[N,M] = size(data); % Number of datapoints. 1201 for mgdata
if M ~= 1
    error('data must be a column vector');
end 

%% Initialise variables needed

t_min = (D-1)*d + 1; % Minimum t index such that XI(t_min) is well defined
t_max = N - L; % Maximum t index such that XO(t_max) is well defined 
N_IO = t_max - t_min + 1; % Number of well defined input/output pairs
XI = nan(N_IO,D); % Preallocate space for XI


%% Fill up XI and XO variables 
% There are faster implementations but this is easy to read
for i = 1:N_IO 
    XI(i,:) = data(i:d:d*(D-1)+i); 
end 
XO = data(t_min+L:end); % Column vector is desired 

end 