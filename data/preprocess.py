import numpy as np

def cross_sectional_norm(data):
    """
    [Arg]
        data: shape of (N_stocks,N_days,N_features)
    [Return]
        data_N: data after cross-sec-norm
    """
    mean_ = np.mean(data,axis=0)
    std_ = np.std(data,axis=0)
    
    data_N = (data-mean_)/std_
    return data_N

def forward_split(data,split_ratio=0.8):
    N_days = data.shape[1]
    split_idx = int(N_days * split_ratio)

    return data[:,:split_idx,:],data[:,split_idx:,:]

def flatten(data):
    """
    [Arg]
        data:shape of (N_stocks,N_days,N_features)
    [Return]
        data:shape of (N_stocks*N_days,N_features)
    """
    N_features = data.shape[2]

    return data.reshape((-1,N_features))

def group_by_stock(data,tickers):
    """
    [Arg]
        data:shape of (N_stocks*N_days,N_features)
    [Return]
        data:shape of (N_stocks,N_days,N_features)
    """
    N_stocks = len(tickers)
    N_features = data.shape[2]

    return data.reshape((N_stocks,-1,N_features))