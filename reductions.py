import numpy as np
from sklearn.feature_selection import RFE

class FSelection:

    def __init__(self):
        raise NotImplementedError

    def get_features(self, X, y, estimator):
        # Does affect the model!
        raise NotImplementedError

class Identity(FSelection):

    def __init__(self):
        print("Applying identity transform! (No transform)")

    def get_features(self, X, y, estimator):
        # Does affect the model!
        return np.arange(X.shape[1])

# class RFEReduction(FSelection):
#
#     def __init__(self):
#         print("Feature selection")
#
#     def get_features(self, X, y, estimator):
#         self.selector = RFE(estimator, n_features_to_select=200, step=1)
#         self.selector = self.selector.fit(X, y)
#         return self.selector.support

