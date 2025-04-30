import numpy as np
from keras.datasets import mnist

from dense import Dense
from convolution import Convolution
from reshape import Reshape
from activations import Sigmoid,Softmax
from losses import cross_entropy , binary_cross_entropy_prime

from network import predict,train

import tensorflow as tf

def one_hot(y):
    res = np.zeros((y.shape[0],10))

    for i in range(y.shape[0]):
        res[i,y[i]] = 1
    return res

def preprocess(X,y,limit1,limit2):
    # zero_index = np.where(y == 0)[0][limit1:limit2]
    # one_index = np.where(y==1)[0][limit1:limit2]
    # all_indices = np.hstack((zero_index,one_index))
    m = X.shape[0]
    all_indices = np.random.permutation(m)
    x,y = X[all_indices],y[all_indices]

    x = x.reshape(len(x),1,28,28)
    x = x.astype("float32") / 255

    y = one_hot(y)
    y = y.reshape(len(y),10,1)

    return x,y

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,y_train = preprocess(x_train,y_train,0,1000)

x_test,y_test = preprocess(x_test,y_test,0,1000)

network = [
    Convolution((1,28,28),3,5),
    Sigmoid(),
    Reshape((5,26,26),(5*26*26,1)),
    Dense(5*26*26,100),
    Sigmoid(),
    Dense(100,10),
    Softmax()
]

train(network,cross_entropy,
      binary_cross_entropy_prime,
      x_train,
      y_train,
      epochs=10,
      learning_rate=0.01)

arc = []

for x,y in zip(x_test,y_test):
    output = predict(network,x)
    if np.argmax(output) == np.argmax(y):
        arc.append(1)
    else:
        pass
print(len(arc) / len(y_test) * 100,"%")
 