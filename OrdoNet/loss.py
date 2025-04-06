class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        # Mean Squared Error: shows how far predictions are from real values
        error = 0  # sum of all errors
        for yt, yp in zip(y_true, y_pred):  # go through each (true, predicted) pair
            error += (yt - yp) ** 2  # add squared difference
        return error / len(y_true)  # average error

    @staticmethod
    def mse_deriv(y_true, y_pred):
        # Derivative of MSE: tells how much to adjust predictions
        n = len(y_true)  # total number of values
        return [(2 * (yp - yt)) / n for yt, yp in zip(y_true, y_pred)]  # gradient for each pair