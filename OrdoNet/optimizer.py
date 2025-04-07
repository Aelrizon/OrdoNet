import math

class AdamOptimizer:
    def __init__(self, size, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr            # learning rate
        self.beta1 = beta1      # retention factor for 1st moment (velocity)
        self.beta2 = beta2      # retention factor for 2nd moment (squared gradients)
        self.eps = eps          # small number to avoid division by zero
        self.t = 0              # step counter
        self.m = [0.0] * size   # first moment, initialized to zeros
        self.v = [0.0] * size   # second moment, initialized to zeros

    def update(self, weights, grads):
        # Check that weights and grads are lists of the same length.
        if len(weights) != len(grads):
            raise ValueError("Weights and gradients must have the same length.")
        
        self.t += 1  # advance training step
        new_weights = []  # list for updated weights
        
        for i in range(len(weights)):
            # Ensure each gradient is a number (convertible to float)
            try:
                grad_i = float(grads[i])
            except Exception as e:
                raise ValueError(f"Gradient at index {i} is not a number: {grads[i]}") from e

            # Update the first moment (velocity): blend old and new gradient
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_i
            # Update the second moment (squared gradient): blend old and new squared gradient
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad_i ** 2)
            
            # Correct bias for the first moment
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Correct bias for the second moment
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Calculate the update step and apply it
            update_val = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
            new_weights.append(weights[i] - update_val)
        
        return new_weights