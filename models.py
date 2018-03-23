import numpy as np
import pandas as pd
#from auto_ml import Predictor
import sys
import os
#from keras.layers import Input, Dense

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
            generations=5,
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

# class Autoencoder(Model):
#
#     def __init__(self):
#         pass
#
#     def build_AE(self, X):
#         ncol = X.shape[1]
#         self.encoding_dim = 200
#         self.input_dim = Input(shape=(ncol,))
#         # DEFINE THE DIMENSION OF ENCODER ASSUMED 3
#         # DEFINE THE ENCODER LAYERS
#         self.encoded1 = Dense(3000, activation='relu')(input_dim)
#         self.encoded2 = Dense(1000, activation='relu')(encoded1)
#         self.encoded3 = Dense(500, activation='relu')(encoded2)
#         self.encoded4 = Dense(self.encoding_dim, activation='relu')(encoded3)
#         # DEFINE THE DECODER LAYERS
#         self.decoded1 = Dense(500, activation='relu')(encoded4)
#         self.decoded2 = Dense(1000, activation='relu')(decoded1)
#         self.decoded3 = Dense(3000, activation='relu')(decoded2)
#         self.decoded4 = Dense(ncol, activation='sigmoid')(decoded3)
#         # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
#         self.autoencoder = Aencoder(input=input_dim, output=decoded4)
#         # CONFIGURE AND TRAIN THE AUTOENCODER
#         self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#         self.autoencoder.fit(X, X, nb_epoch=100, batch_size=100, shuffle=True)
#         # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
#
#     def transform_AE(X):
#         self.encoder = Aencoder(input = self.input_dim, output = self.encoded4)
#         self.encoded_input = Input(shape = (self.encoding_dim, ))
#         return self.encoder.predict(X_test)
#
#     def fit(self, X, y):
#         """ The autoencoder will be trained here! """
#         pass
#
#
#     def predict(self, X):
#         pass
#
#     def transform(self, X, y):
#         pass

