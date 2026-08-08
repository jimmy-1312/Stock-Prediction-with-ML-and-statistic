for LSTM:

Input: already to gpu tensor, shape of (B,L,data_dim)
L = window_size, preset is 5.
data_dim = 1 when testing

Output: shape of (B,1)
same in price and log_return since both just one value.