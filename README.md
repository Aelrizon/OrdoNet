# OrdoNet

A complete neural network library built from scratch — no frameworks.

---

## Purpose

To build a simple but working neural network system  
that shows the full process: from raw data to prediction.

This is not a toy or a training script —  
it’s a full engineering project with:

- clean architecture  
- modular code  
- training and prediction  
- saving and loading  
- CSV input and visualization

---

## What's inside

OrdoNet:
  - `matrix.py` — basic math operations (dot, scalar, etc.)
  - `activation.py` — activation functions: sigmoid, tanh, relu, softmax
  - `loss.py` — loss function (MSE) and its derivative
  - `optimizer.py` — Adam optimizer, implemented from scratch
  - `neuron.py` — a single neuron with forward/backward logic
  - `layer.py` — a full layer of neurons
  - `network.py` — training, prediction, save/load
  - `dataset.py` — load CSV, normalize, create mini-batches
  - `utils.py` — logging, progress bar, loss visualization
Example:
  — code examples
Doc = Documentation for the library.

---

## Quick start

```python
from network import Network

net = Network([4, 5, 1])
net.train(X, y, epochs=100, lr=0.01)
