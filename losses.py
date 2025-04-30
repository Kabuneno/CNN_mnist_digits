import numpy as np



def cross_entropy(y_true,y_pred):
    return np.mean(-np.log(y_pred+1e-9)* y_true )

def binary_cross_entropy_prime(y_true,y_pred):
    return (y_pred- y_true) / y_true.shape[0]

