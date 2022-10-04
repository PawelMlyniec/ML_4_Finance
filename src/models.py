import numpy as np
from sklearn.dummy import DummyRegressor


class LastValueRegressor(DummyRegressor):
    def __init__(self, win_size):
        super().__init__()
        self.win_size = win_size

    def predict(self, X):
        """
        Dummy predict that predicts
        the price from the last timestep it saw
        """
        if len(X) == 1:
            return np.array([X[0][-1]])
        return X[:, -1]
