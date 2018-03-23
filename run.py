from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np

from importer import Importer
from models import RandomModel

class Runner():

    def __init__(self, verbose=False):
        print("Starting runer")
        self.imp = Importer()
        self.X_train, self.y_H, self.y_I, self.y_M = self.imp.get_X_y()

        if verbose:
            print("Printing imported shapes", self.X_train.shape, self.y_H.shape, self.y_I.shape, self.y_M.shape)

    def run(self, model, verbose=True):
        """
            Selects the best mode
        """
        print("Run!")

        f1_I = []
        f1_H = []
        f1_M = []

        kf = KFold(n_splits=10, shuffle=False)
        for train_index, test_index in kf.split(self.X_train):

            X_train = self.X_train[train_index]
            y_I_train = self.y_I[train_index]
            y_H_train = self.y_H[train_index]
            y_M_train = self.y_M[train_index]

            X_cv = self.X_train
            y_I_cv = self.y_I[test_index]
            y_H_cv = self.y_H[test_index]
            y_M_cv = self.y_M[test_index]

            if verbose:
                print("Printing all shapes:", X_train.shape, y_I_train.shape, y_H_train.shape, y_M_train.shape)
                print("CV", X_cv.shape, y_I_cv.shape, y_H_cv.shape, y_M_cv.shape )

            # Go through each array!
            # I first!
            model.fit(X_train, y_I_train)
            y_pred = model.predict(X_train, y_I_cv)
            f1_value = f1_score(y_pred, y_I_cv)
            f1_I.append(f1_value)

            # H second!
            model.fit(X_train, y_H_train)
            y_pred = model.predict(X_train, y_H_cv)
            f1_value = f1_score(y_pred, y_H_cv)
            f1_H.append(f1_value)

            # M third!
            model.fit(X_train, y_M_train)
            y_pred = model.predict(X_train, y_M_cv)
            f1_value = f1_score(y_pred, y_M_cv)
            f1_M.append(f1_value)

        print("Predicted average f1 scores are: ")
        print("F1 score for I: ", np.mean(f1_I))
        print("F1 score for M: ", np.mean(f1_M))
        print("F1 score for H: ", np.mean(f1_H))

        #
        #     # classifier with ensemble
        #     # ATTENTION: unbalanced data
        #     # adaboost for each of the problem
        #     # reweighting (later)
        #     # can play with K
        #     clf.fit(X_M_train, y_M_train)
        #     score = f1(X_M_validation, y_M_validation)
        #     print("Score " + score)

    def get_baseline(self, X):
        # This baseline is calculated doing random predictions
        model = RandomModel()
        y_hat = model.predict(X)

    def batch_run_models(self, arr_models):

        for model in arr_models:
            pass




if __name__ == "__main__":
    print("Starting runner!")
    #imp = Importer()
    #X_train, y_M, y_I, y_H = imp.get_X_y()
    # print(X_train.head(5))
    # print(y_M.head(5))
    # print(y_I.head(5))
    # print(y_H.head(5))

    runner = Runner()
    rndModel = RandomModel
    runner.run(rndModel)