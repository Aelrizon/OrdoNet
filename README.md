# OrdoNet

A simple yet complete neural network library built entirely from scratch — no external deep learning frameworks.

## Overview

**OrdoNet** is an educational and demonstration-oriented neural network library that showcases the core concepts of neural networks:

- Forward and backward passes  
- Basic activation functions (Sigmoid, Tanh, ReLU, Softmax)  
- A simple feed-forward architecture with fully-connected layers  
- A custom implementation of backpropagation  
- An example of an optimizer (Adam)

The main goal is to give students and enthusiasts a clear view of how a neural network operates under the hood, without abstracting away important implementation details.

## Features

- **Layer-by-layer design:** `Layer` and `Neuron` classes manage forward and backward propagation.  
- **Custom activation functions:** sigmoid, tanh, relu, and softmax (with derivatives).  
- **Loss functions:** currently implements Mean Squared Error (MSE) and its derivative.  
- **Optimizers:** Adam optimizer is available for parameter updates.  
- **Data handling:** includes simple CSV loading, normalization, and batching.  
- **Utilities:** logging with timestamps, progress bar for training loops, and optional matplotlib-based loss plotting.

## Project Structure

```
├── activation.py   # Activation functions and their derivatives
├── dataset.py      # CSV loading, normalization, and batch generation
├── layer.py        # Layer class that holds multiple neurons
├── loss.py         # Loss function (MSE) and its derivative
├── matrix.py       # Basic matrix operations (not heavily used in the main code yet)
├── network.py      # Network class orchestrating layers, forward/backward passes
├── neuron.py       # Neuron class with weights, bias, and activation
├── optimizer.py    # AdamOptimizer class for parameter updates
├── utils.py        # Utility functions: logging, progress bar, plotting
└── Example         # Example scripts showing how to use the library
```

## Installation

No special installation is required for OrdoNet itself; just clone the repository and ensure you have Python 3.x. Optionally, install [matplotlib](https://matplotlib.org) if you want to visualize the training loss:

```bash
pip install matplotlib
```

## Quick Start

Below is a minimal example demonstrating how to build and train a small network using OrdoNet:

```python
from network import Network
from loss import Loss
from optimizer import AdamOptimizer

# Define the architecture: 2 inputs -> 3 hidden neurons -> 1 output
net = Network([2, 3, 1])

# Example training data (XOR-like)
data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
targets = [
    [0],
    [1],
    [1],
    [0]
]

# Create an Adam optimizer for the network
adam = AdamOptimizer(size=net.total_parameters(), lr=0.01)

epochs = 1000
loss_history = []

for epoch in range(epochs):
    total_loss = 0
    for X, y in zip(data, targets):
        prediction = net.forward(X)
        loss = Loss.mse(y, prediction)
        total_loss += loss
        grads = net.backward(y)
        params = net.get_parameters()
        new_params = adam.update(params, grads)
        net.set_parameters(new_params)
    loss_history.append(total_loss / len(data))

print("Training complete!")
print("Final average loss:", loss_history[-1])

for X in data:
    print("Input:", X, "Prediction:", net.predict(X))
```

## Examples

Check out the `Example` folder in this repository for:

- **Salary Prediction (`Salary.py`)** — Simple regression example predicting salary based on years of experience  
- **Sine Wave Regression (`Sine_Wave_Regression.py`)** — Fits a sine function with a single hidden layer  
- **XOR Problem (`XOR.py`)** — Classic XOR demonstration with a small feed-forward network

These scripts illustrate how to load data, train the network, and visualize the loss.

## Next Steps & Possible Improvements

- Add more activation and loss functions (e.g., Leaky ReLU, ELU, Cross-Entropy)  
- Use NumPy or PyTorch tensors for faster matrix operations  
- Implement regularization (dropout, L1/L2)  
- Add unit tests for stability and correctness  
- Expand documentation and create formal API reference

## License

```
MIT License  
Copyright (c) 2025  
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

Feel free to fork, experiment, and contribute 💡
