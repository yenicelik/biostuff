import numpy as np
import pandas as pd
from auto_ml import Predictor

from sklearn.ensemble import AdaBoostClassifier

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
        print("Starting Random model as a baseline!")

    def predict(self, X):
        # Return a random prediction for each sample
        return np.random.randint(2, size=X.shape[0])

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass

class AutoML(Model):

    def __init__(self):
        print("Starting AutoML")

        self.parameter1 = 0.


    def predict(self, X):
        return self.trained_model.predict(X)

    def fit(self, X, y):

        # Rehsape it into a pandas dataframe, and re-write the model and column_description!
        df_train = pd.DataFrame({'X': X, 'Y': y})

        self.column_descriptions = {
            'X': 'numerical',
            'Y': 'output'
        }

        self.ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=self.column_descriptions)

        self.ml_predictor.fit(df_train)

    def transform(self, X, y):
        pass

class AdaBoostModel(Model):

    def __init__(self):
        print("Performing Adaboost!")

    def fit(self, X, y):
        self.model = AdaBoostClassifier(n_estimators=10, learning_rate=1.)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def transform(self, X, y):
        pass

