import random
import math

class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights for each input with random values from -1 to 1.
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        # Initialize bias with a random value.
        self.bias = random.uniform(-1, 1)
        # These will store inputs and the weighted sum for backpropagation.
        self.last_input = None   # Saved input vector.
        self.last_z = None       # Saved weighted sum before activation.
        self.output = None       # Output after activation.

    def forward(self, inputs):
        # Save input for backpropagation.
        self.last_input = inputs
        # Compute weighted sum: (weight * input) for each input plus bias.
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.last_z = z
        # Apply sigmoid activation function to get the output.
        self.output = 1 / (1 + math.exp(-z))  # Can be replaced with Activation.sigmoid(z).
        return self.output

    def backward(self, d_output):
        # d_output is the gradient of the loss with respect to the neuron's output.
        # If d_output is a list, extract its first element.
        if isinstance(d_output, list):
            d_output = d_output[0]
        # Compute the sigmoid derivative at the stored weighted sum.
        s = 1 / (1 + math.exp(-self.last_z))  # Same as sigmoid(self.last_z)
        d_activation = s * (1 - s)            # Sigmoid derivative: s * (1 - s)
        # Calculate delta: how much to adjust the weighted sum.
        delta = d_output * d_activation
        # Compute gradients for each weight: delta multiplied by the corresponding input.
        grad_weights = [delta * x for x in self.last_input]
        # The gradient for the bias is just delta.
        grad_bias = delta
        # Compute gradients with respect to inputs to propagate to previous layers.
        d_inputs = [delta * w for w in self.weights]
        return grad_weights, grad_bias, d_inputs

    def update(self, grad_weights, grad_bias, lr):
        # Update weights using the gradients and learning rate.
        self.weights = [w - lr * gw for w, gw in zip(self.weights, grad_weights)]
        # Update bias similarly.
        self.bias -= lr * grad_bias