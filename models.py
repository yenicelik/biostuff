import numpy as np
import pandas as pd
#from auto_ml import Predictor
import sys
import os

from tpot import TPOTClassifier

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


class TPot(Model):

    def __init__(self):
        print("Starting t pot!")

    def fit(self, X, y, title=None):
        # For this case, X and y are the complete datasets!!!
        self.pipeline_optimizer = TPOTClassifier(
            generations=1, #5,
            cv=5,
            random_state=42,
            verbosity=3,
            n_jobs=8,
            max_eval_time_mins=1,#10,
            scoring='f1',
            subsample=0.5
        )
        self.pipeline_optimizer.fit(X, y)

        if not os.path.exists("./automl"):
            os.makedirs("./automl")

        self.pipeline_optimizer.export('./automl/tpot_exported_pipeline_' + str(title) + '_.py')

    def predict(self, X):
        pass

