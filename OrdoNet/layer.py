from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_inputs):
        # create a layer as a list of neurons
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        # calculate outputs for all neurons in the layer
        self.inputs = inputs  # save inputs for backprop
        return [neuron.forward(inputs) for neuron in self.neurons]

    def backward(self, d_outputs):
        # compute gradients for each neuron using output errors
        grads = [neuron.backward(d_out) for neuron, d_out in zip(self.neurons, d_outputs)]
        # split into weight gradients and bias gradients
        grad_weights, grad_biases = zip(*grads)
        return list(grad_weights), list(grad_biases)

    def update_weights(self, lr, grad_weights_all, grad_biases_all):
        # update weights and biases for each neuron
        for neuron, grad_w, grad_b in zip(self.neurons, grad_weights_all, grad_biases_all):
            neuron.update(grad_w, grad_b, lr)