import math

class Activation:
    @staticmethod
    def sigmoid(x):
        """
        Computes the sigmoid of x.
        Squeezes a single number between 0 and 1.
        """
        try:
            return 1 / (1 + math.exp(-x))
        except Exception as e:
            raise ValueError(f"Error computing sigmoid for x={x}: {e}")

    @staticmethod
    def sigmoid_deriv(x):
        """
        Computes the derivative of the sigmoid function at x.
        This is used to determine how much to adjust during training.
        """
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        """
        Computes the hyperbolic tangent of x.
        Squeezes a single number between -1 and 1.
        """
        try:
            return math.tanh(x)
        except Exception as e:
            raise ValueError(f"Error computing tanh for x={x}: {e}")

    @staticmethod
    def tanh_deriv(x):
        """
        Computes the derivative of tanh at x.
        Used during backpropagation to adjust weights.
        """
        try:
            t = math.tanh(x)
            return 1 - t * t
        except Exception as e:
            raise ValueError(f"Error computing tanh derivative for x={x}: {e}")

    @staticmethod
    def relu(x):
        """
        Applies the ReLU function: returns x if x > 0, else returns 0.
        """
        try:
            return x if x > 0 else 0
        except Exception as e:
            raise ValueError(f"Error computing relu for x={x}: {e}")

    @staticmethod
    def relu_deriv(x):
        """
        Computes the derivative of ReLU: returns 1 if x > 0, else 0.
        """
        try:
            return 1 if x > 0 else 0
        except Exception as e:
            raise ValueError(f"Error computing relu derivative for x={x}: {e}")

    @staticmethod
    def softmax(vector):
        """
        Computes the softmax function for a list of numbers.
        Turns the numbers into probabilities that sum to 1.
        """
        if not isinstance(vector, (list, tuple)):
            raise TypeError("Input to softmax must be a list or tuple.")
        if len(vector) == 0:
            raise ValueError("Input vector for softmax cannot be empty.")
        
        try:
            # For numerical stability, subtract the maximum value
            biggest = max(vector)
            powered = [math.exp(x - biggest) for x in vector]
            total = sum(powered)
            if total == 0:
                raise ValueError("Sum of exponentials in softmax is zero.")
            return [x / total for x in powered]
        except Exception as e:
            raise ValueError(f"Error computing softmax for vector={vector}: {e}")