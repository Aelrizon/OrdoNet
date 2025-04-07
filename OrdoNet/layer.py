from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_inputs):
        # Create a layer as a list of neurons.
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        # Calculate outputs for all neurons in the layer.
        self.inputs = inputs  # Save inputs for backpropagation.
        return [neuron.forward(inputs) for neuron in self.neurons]

    def backward(self, d_outputs):
        """
        Computes gradients for each neuron using output errors and aggregates input gradients.
        
        d_outputs: list of error gradients for each neuron in this layer.
        Returns:
          - grad_weights_all: list of weight gradients for each neuron.
          - grad_biases_all: list of bias gradients for each neuron.
          - d_inputs: aggregated gradients with respect to inputs (to propagate to previous layer).
        """
        # Compute gradients for each neuron.
        # Each neuron.backward returns: (grad_weights, grad_bias, d_inputs)
        grads = [neuron.backward(d_out) for neuron, d_out in zip(self.neurons, d_outputs)]
        grad_weights_all, grad_biases_all, d_inputs_list = zip(*grads)
        
        # Aggregate input gradients: sum d_inputs from each neuron element-wise.
        d_inputs = [0.0 for _ in range(len(self.inputs))]
        for d in d_inputs_list:
            for j, val in enumerate(d):
                d_inputs[j] += val

        return list(grad_weights_all), list(grad_biases_all), d_inputs

    def update_weights(self, lr, grad_weights_all, grad_biases_all):
        # Update weights and biases for each neuron using given gradients and learning rate.
        for neuron, grad_w, grad_b in zip(self.neurons, grad_weights_all, grad_biases_all):
            neuron.update(grad_w, grad_b, lr)