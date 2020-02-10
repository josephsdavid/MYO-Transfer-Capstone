# Preprocessing and Input Notes  

## Inputs

### Input Shape

#### ConvLSTM layer input

> ConvLSTM layer input
The LSTM cell input is a set of data over time, that is, a 3D tensor with shape (samples, time_steps, features). The Convolution layer input is a set of images as a 4D tensor with shape (samples, channels, rows, cols). The input of a ConvLSTM is a set of images over time as a 5D tensor with shape (samples, time_steps, channels, rows, cols).

_src: [https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7]()_


> The input shape for an LSTM must be (num_samples, num_time_steps, num_features). In your example case, combining both cities as input, num_features will be 2x3=6.  
>
>If you lump all your 365 time steps into one sample, then the first dimension will be 1 - one single sample! You can also do sanity check by using the total number of data points. You have 2 cities, each with 365 time-steps and 3 features: 2x365x3= 2190 . This is obviously the same as 1x365x6 (as I said above) - so it would be a possibility (Keras will run) - but it obviously won't learn to generalise at all, only giving it one sample.  

_src: [https://datascience.stackexchange.com/questions/27563/multi-dimentional-and-multivariate-time-series-forecast-rnn-lstm-keras/27572#27572]()_