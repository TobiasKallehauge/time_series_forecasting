# Time series forecasting
Matlab library for time series forecasting using feed forward neural networks (FNN's) and fuzzy logic networks (NF). The library is build to predict the [Mackley glass](https://ww2.mathworks.cn/help/fuzzy/predict-chaotic-time-series-code.html) timeseries, but any 1 dimensional timeseries can be used. 

## About (theory)
The feed forward neural network (or multilayer perceptron) supports 4 different acivation functions: Hyperbolic tangent, Sigmoid, ReLU and the identity. Any number of layers and nodes in each layer can be initialised. Only fully connected networks can be realised. See Pattern Recognition and Machine learning by Bishop,Christopher M. page 227-246 for more information.

The fuzzy logic network is a Takagi-Sugeno type Fuzzy model that uses fuzzy rules with Gaussian membership functions (GMF) to determine the degree of fulfillement. The output for each rule is an affine function of the input and the final output is computed as a weigthed sum using the degree of fulfillment. See Compitational Intelligence in Time Series Forecasting by Palit, Ajoy K and Popovic, Dobrivoje, 2015 chapter 4 and 6 for more information. 

The library was developed as a part of the course "Computational Intelligence in Modelling, Prediction and Signal Processing" by Palit, A. K. The written report associated with this report is included in the liberary as "Predicting the Makley Glass time series.pdf"

## The liberay
The liberay is solely built on the basic tools in matlab - no toolboxes required. All functions are well documented and explains the input, parameters and output. Type help to see the documentation for any function. The main files are inteded as an examle for how to used the library. 
### Contents
* **TK_main_FNN.m** - Example of how to fit a feed FFN to a timeseries. The Mackley glass timeseries is used here as an example but other types can be used. 
* **TK_FNN_forward.m** - Forward propagation of the FNN. 
* **TK_FNN_forward_vec.m** - Call the FNN for multible inputs (each input may be more than one sample).
* **TK_FNN_grad.m** - Compute the gradient of the FNN with respect to half sum of squared error using backpropagation.
* **TK_FNN_init_para.m** - Initialise the parameterss for the FNN. Weights are initialised uniformly between 0 and 1 and the biasses are initialsed as zero.
* **TK_NF_forward.m** - Inference of the NF network. Note that only single ouput is supported. 
* **TK_NF_forward_vec.m** - Inference of the NF network over multible inputs (each input may be more than one sample).
* TK_NF_grad.m Compute the gradient of the NF network with respect to half sum of squared error.
* **TK_FNN_init_para.m** - Initialise the parameterss for the NF network. Center parameters for the GMF's are initialised uniformly on the range of the time series while the variances and rule weight parameters are initialised uniformly between 0 and 1.
* **TK_optimize.m** - Train either FNN or NF network using gradient descent type optimization. The gradient step can be done with respect to one ore multible training examples determined by the given batch size which defaults to 1. Serveral stopping criteria are supported (see the documentation), and a graphical progress bar is shown during optimization. 
* **TK_timeseries_preprocess.m** - Preprocess vector containing a time timeseries returning XI and XO. XI have inputs along the rows with one or more columns. The inputs may be distanced by one or more samples. XO is a column vector with the output used as target for prediction. 

### Versioning
Matlab version: '9.6.0.1174912 (R2019a) Update 5'

## Authors
[Tobias Kallehauge](https://www.linkedin.com/in/tobias-kallehauge/) - MS student - Aalborg Univeristy

## License
The content of this library is freely available.

## Acknowledgments
Thank you to Professor Ajoy. K. Palit who gave this exercise and guided the process. 
