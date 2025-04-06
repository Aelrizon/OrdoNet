import random
import math

class Neuron:
    def __init__(self, num_inputs):
        # Init weights for each input with random values from -1 to 1
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        # Init bias with a random value
        self.bias = random.uniform(-1, 1)
        # These will store inputs and internal values for backprop
        self.last_input = None   # saves input vector
        self.last_z = None       # saves weighted sum before activation
        self.output = None       # output after activation

    def forward(self, inputs):
        # Save input for backpropagation
        self.last_input = inputs
        # Weighted sum: sum of (weight * input) + bias
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.last_z = z
        # Apply activation (e.g. sigmoid) to get output
        self.output = 1 / (1 + math.exp(-z))  # can replace with Activation.sigmoid(z)
        return self.output

    def backward(self, d_output):
        # d_output is the loss gradient w.r.t. the neuron output
        # Compute sigmoid derivative at last_z
        s = 1 / (1 + math.exp(-self.last_z))  # same as sigmoid(self.last_z)
        d_activation = s * (1 - s)            # sigmoid derivative: s * (1 - s)
        # Delta = how much to change the weighted sum
        delta = d_output * d_activation
        # Gradient for each weight: delta * input
        grad_weights = [delta * x for x in self.last_input]
        # Gradient for bias = just delta
        grad_bias = delta
        return grad_weights, grad_bias

    def update(self, grad_weights, grad_bias, lr):
        # Update weights using learning rate and gradients
        self.weights = [w - lr * gw for w, gw in zip(self.weights, grad_weights)]
        # Update bias
        self.bias -= lr * grad_bias