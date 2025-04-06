import math

class AdamOptimizer:
    def __init__(self, size, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr            # learning rate
        self.beta1 = beta1      # how much to keep past speed (1st moment)
        self.beta2 = beta2      # how much to keep past spread (2nd moment)
        self.eps = eps          # tiny value to avoid division by zero
        self.t = 0              # step counter
        self.m = [0] * size     # first moment (like velocity), starts at 0
        self.v = [0] * size     # second moment (like energy), starts at 0

    def update(self, weights, grads):
        self.t += 1  # next training step
        new_weights = []  # updated weights go here
        for i in range(len(weights)):
            # update speed (1st moment): blend old and new
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            # update spread (2nd moment): blend old and new squared
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            # fix bias in 1st moment (early steps are too low)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # fix bias in 2nd moment
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # calculate step and apply it
            new_weights.append(weights[i] - self.lr * m_hat / (math.sqrt(v_hat) + self.eps))
        return new_weights