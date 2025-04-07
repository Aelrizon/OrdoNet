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
        # Create each layer (skip input layer)
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(num_neurons=layer_sizes[i],
                                     num_inputs=layer_sizes[i - 1]))

    def forward(self, inputs, debug=False):
        """
        Runs the data through the network (forward pass).

        inputs: input vector (like image features)
        debug: if True, prints the output of each layer.
        """
        data = inputs  # start with input
        for index, layer in enumerate(self.layers):
            data = layer.forward(data)  # pass through layer
            if debug:
                print(f"Layer {index+1} output: {data}")  # print layer output
        return data  # final prediction

    def backward(self, target):
        """
        Runs backpropagation through the network and returns a flattened list
        of parameter gradients.

        target: correct output (what the network should predict)
        """
        # Get prediction from output layer (assumes first layer saved its inputs)
        output = self.forward(self.layers[0].inputs)
        # Calculate loss gradient for the output (returns a list for output layer)
        d_outputs = Loss.mse_deriv(target, output)
        parameter_gradients = []  # flattened gradients for all layers
        # Propagate gradients backwards through each layer
        for layer in reversed(self.layers):
            # layer.backward now returns: (grad_weights_list, grad_biases_list, d_inputs)
            grad_weights_list, grad_biases_list, d_inputs = layer.backward(d_outputs)
            # Flatten gradients for this layer
            for gw, gb in zip(grad_weights_list, grad_biases_list):
                parameter_gradients.extend(gw)
                parameter_gradients.append(gb)
            # Use d_inputs as the gradient to pass to the previous layer
            d_outputs = d_inputs
        return parameter_gradients

    def update(self, lr):
        """
        Updates weights in each layer.

        lr: learning rate — how much to adjust weights.
        """
        for layer in self.layers:
            layer.update_weights(lr)

    def train(self, data, targets, epochs, lr):
        """
        Trains the network on data.

        data: list of input vectors.
        targets: list of correct outputs.
        epochs: number of full passes over the dataset.
        lr: learning rate.
        """
        for epoch in range(epochs):
            total_loss = 0  # total error for this epoch
            for inputs, target in zip(data, targets):
                output = self.forward(inputs)  # forward pass
                loss = Loss.mse(target, output)  # calculate error
                total_loss += loss             # accumulate error
                # Compute parameter gradients via backpropagation
                grads = self.backward(target)
                # Get current parameters (flattened)
                params = self.get_parameters()
                # Update parameters using the optimizer externally (Adam, etc.)
                # Here, we assume the optimizer call happens outside and parameters are set
                # For now, we do a simple update using our update() method
                self.update(lr)
            avg_loss = total_loss / len(data)  # average loss
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

    def predict(self, inputs):
        """
        Gets prediction for the given input.
        """
        return self.forward(inputs)

    def save(self, filename):
        """
        Saves model weights and biases to a file.
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
        Loads model weights and biases from a file.
        """
        with open(filename, 'r') as f:
            model_data = eval(f.read())
        for layer, layer_data in zip(self.layers, model_data):
            for neuron, neuron_data in zip(layer.neurons, layer_data):
                neuron.weights = neuron_data['weights']
                neuron.bias = neuron_data['bias']

    def total_parameters(self):
        """
        Returns the total number of parameters (weights + biases) in the whole network.
        Needed to initialize the optimizer.
        """
        total = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                total += len(neuron.weights) + 1  # +1 for the bias
        return total

    def get_parameters(self):
        """
        Returns all network parameters (weights and biases) as a flattened list.
        """
        params = []
        for layer in self.layers:
            for neuron in layer.neurons:
                params.extend(neuron.weights)
                params.append(neuron.bias)
        return params

    def set_parameters(self, new_params):
        """
        Sets network parameters from a flattened list.
        """
        index = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                num_weights = len(neuron.weights)
                neuron.weights = new_params[index:index + num_weights]
                index += num_weights
                neuron.bias = new_params[index]
                index += 1