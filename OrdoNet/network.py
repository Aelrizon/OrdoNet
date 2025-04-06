from layer import Layer  # import Layer class (contains neurons)
from loss import Loss    # import loss functions

class Network:
    def __init__(self, layer_sizes):
        """
        Builds a neural network.

        layer_sizes: list of how many neurons in each layer.
        Example: [2, 4, 1] means:
            - 2 input values
            - 1 hidden layer with 4 neurons
            - 1 output neuron
        """
        self.layers = []  # stores all layers in the network

        # for each layer (skip input layer)
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(num_neurons=layer_sizes[i],
                                     num_inputs=layer_sizes[i - 1]))

    def forward(self, inputs, debug=False):
        """
        Runs the data through the network (forward pass)

        inputs: input vector (like image features)
        debug: if True, print output of each layer
        """
        data = inputs  # start with input
        for index, layer in enumerate(self.layers):
            data = layer.forward(data)  # pass through layer
            if debug:
                print(f"Layer {index+1} output: {data}")  # print layer output
        return data  # final prediction

    def backward(self, target):
        """
        Runs backpropagation.

        target: correct output (what the network should predict)
        """
        output = self.forward(self.layers[0].inputs)  # get prediction
        grad = Loss.mse_deriv(target, output)         # calculate error gradient
        for layer in reversed(self.layers):           # go backwards through layers
            grad = layer.backward(grad)
        return grad

    def update(self, lr):
        """
        Updates weights in each layer.

        lr: learning rate — how much to adjust weights
        """
        for layer in self.layers:
            layer.update_weights(lr)

    def train(self, data, targets, epochs, lr):
        """
        Trains the network on data.

        data: list of input vectors
        targets: list of correct outputs
        epochs: how many times to go over the dataset
        lr: learning rate
        """
        for epoch in range(epochs):
            total_loss = 0  # total error for this epoch
            for inputs, target in zip(data, targets):
                output = self.forward(inputs)         # run forward
                loss = Loss.mse(target, output)       # get error
                total_loss += loss                    # accumulate error
                self.backward(target)                 # backprop
                self.update(lr)                       # update weights
            avg_loss = total_loss / len(data)         # average loss
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

    def predict(self, inputs):
        """
        Gets prediction for input.
        """
        return self.forward(inputs)

    def save(self, filename):
        """
        Saves model weights and biases to file.
        """
        model_data = []
        for layer in self.layers:
            layer_data = []
            for neuron in layer.neurons:
                layer_data.append({
                    'weights': neuron.weights,
                    'bias': neuron.bias
                })
            model_data.append(layer_data)
        with open(filename, 'w') as f:
            f.write(str(model_data))

    def load(self, filename):
        """
        Loads weights and biases from file.
        """
        with open(filename, 'r') as f:
            model_data = eval(f.read())
        for layer, layer_data in zip(self.layers, model_data):
            for neuron, neuron_data in zip(layer.neurons, layer_data):
                neuron.weights = neuron_data['weights']
                neuron.bias = neuron_data['bias']