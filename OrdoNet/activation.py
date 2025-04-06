import math

class Activation:
    @staticmethod
    def sigmoid(x):
        # Squeezes value between 0 and 1
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        # How much to change (for training)
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        # Squeezes value between -1 and 1
        return math.tanh(x)

    @staticmethod
    def tanh_deriv(x):
        # Derivative of tanh (for training)
        t = math.tanh(x)
        return 1 - t * t

    @staticmethod
    def relu(x):
        # Keeps positive values, cuts negatives
        return x if x > 0 else 0

    @staticmethod
    def relu_deriv(x):
        # 1 if x is positive, else 0
        return 1 if x > 0 else 0

    @staticmethod
    def softmax(vector):
        # Turns numbers into probabilities
        biggest = max(vector)                              # for stability
        powered = [math.exp(x - biggest) for x in vector]  # exponentiate each
        total = sum(powered)                               # sum all
        return [x / total for x in powered]                # divide to normalize