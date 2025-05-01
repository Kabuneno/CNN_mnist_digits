# Neural Network Framework with CNN Implementation

This project implements a neural network framework from scratch in Python with a focus on Convolutional Neural Networks (CNNs). The framework includes various layer types, activation functions, and a demonstration of image classification using the MNIST dataset.

## Project Structure

- **layer.py**: Base class for all neural network layers
- **activation.py**: Base class for activation functions
- **activations.py**: Implementation of Sigmoid and Softmax activation functions
- **dense.py**: Implementation of fully connected (dense) layer
- **convolution.py**: Implementation of convolutional layer
- **reshape.py**: Layer for reshaping tensors between convolution and dense layers
- **losses.py**: Loss functions implementation (cross-entropy)
- **network.py**: Core functions for network training and prediction
- **mnist_conv.py**: Example application using the framework on MNIST

## Features

- Modular layer architecture
- Support for convolutional neural networks
- Forward and backward propagation implementation for all layers
- Gradient descent optimization
- MNIST digit classification example

## Requirements

- Python 3.x
- NumPy
- SciPy
- Keras (for loading the MNIST dataset)
- TensorFlow (dependency for Keras)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cnn-project.git
cd cnn-project

# Install required packages
pip install numpy scipy keras tensorflow
```

## Usage

### Basic Example

```python
from dense import Dense
from activations import Sigmoid, Softmax
from losses import cross_entropy, binary_cross_entropy_prime
from network import predict, train

# Create a simple network
network = [
    Dense(input_size=784, output_size=100),
    Sigmoid(),
    Dense(100, 10),
    Softmax()
]

# Train the network
train(network, cross_entropy, binary_cross_entropy_prime, 
      x_train, y_train, epochs=10, learning_rate=0.1)

# Make predictions
prediction = predict(network, x_test[0])
```

### CNN Example (MNIST)

Run the MNIST example directly:

```bash
python mnist_conv.py
```

This will:
1. Load the MNIST dataset
2. Preprocess the data
3. Create a CNN with the architecture:
   - Convolutional layer (5 filters of size 3x3)
   - Sigmoid activation
   - Reshape layer
   - Dense layer (to 100 neurons)
   - Sigmoid activation
   - Dense layer (to 10 neurons)
   - Softmax activation
4. Train the network for 10 epochs
5. Evaluate and print accuracy on test data

## Implementation Details

### Layers

- **Layer**: Abstract base class with forward and backward methods
- **Dense**: Fully connected layer with weights and biases
- **Convolution**: 2D convolutional layer implementation
- **Reshape**: Layer to reshape data between convolutional and dense layers
- **Activation**: Base class for activation functions
- **Sigmoid**: Sigmoid activation function
- **Softmax**: Softmax activation for classification outputs

### Training

The `network.py` module provides two main functions:
- `predict()`: Forward pass through the network
- `train()`: Training loop with gradient descent
