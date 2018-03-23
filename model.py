import numpy as np

class Model:

    def __init__(self):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X, y):
        raise NotImplementedError

class RandomModel(Model):

    def __init__(self):
        print("Starting Random model as a baselin!")

    def predict(self, X):
        # Return a random prediction for each sample
        return np.random.rand(X.shape[0])

    def fit(self, X):
        pass

    def transform(self, X):
        pass