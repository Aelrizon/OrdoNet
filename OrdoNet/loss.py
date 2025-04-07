class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        """
        Mean Squared Error: measures how far predictions are from true values.
        
        Both y_true and y_pred should be lists of numbers of equal length.
        """
        if len(y_true) == 0:
            raise ValueError("y_true is empty.")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        error = 0.0  # sum of squared differences
        for yt, yp in zip(y_true, y_pred):
            error += (yt - yp) ** 2  # add squared difference
        return error / len(y_true)  # average error

    @staticmethod
    def mse_deriv(y_true, y_pred):
        """
        Derivative of MSE: gives the gradient for each (true, predicted) pair.
        
        Both y_true and y_pred should be lists of numbers of equal length.
        """
        if len(y_true) == 0:
            raise ValueError("y_true is empty.")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        n = len(y_true)  # total number of values
        # Calculate gradient for each pair: (2 * (predicted - true)) / n
        return [(2 * (yp - yt)) / n for yt, yp in zip(y_true, y_pred)]