from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np

from importer import Importer
from models import RandomModel, AdaBoostModel, TPot
from reductions import Identity, RFEReduction

from sklearn.ensemble import AdaBoostClassifier

class Runner():

    def __init__(self, verbose=True):
        print("Starting runer")
        self.imp = Importer()
        self.X_train, self.y_H, self.y_I, self.y_M = self.imp.get_X_y()

        if verbose:
            print("Printing imported shapes", self.X_train.shape, self.y_H.shape, self.y_I.shape, self.y_M.shape)
            print("Printing imported shapes", self.X_train[:3], self.y_H[:3], self.y_I[:3], self.y_M[:3])

    def run(self, reduction, model, verbose=False):
        """
            Selects the best mode
        """
        print("Run!")

        f1_I = []
        f1_H = []
        f1_M = []

        i = 0

        kf = KFold(n_splits=10, shuffle=False)
        for train_index, test_index in kf.split(self.X_train):
            i += 1
            print("Progress: " + str(i) + " out of " + str(10))

            X_train = self.X_train[train_index]
            y_I_train = self.y_I[train_index]
            y_H_train = self.y_H[train_index]
            y_M_train = self.y_M[train_index]

            X_cv = self.X_train[test_index]
            y_I_cv = self.y_I[test_index]
            y_H_cv = self.y_H[test_index]
            y_M_cv = self.y_M[test_index]

            if verbose:
                print("Printing all shapes:", X_train.shape, y_I_train.shape, y_H_train.shape, y_M_train.shape)
                print("CV", X_cv.shape, y_I_cv.shape, y_H_cv.shape, y_M_cv.shape )

            assert(test_index.shape[0] > 0)

            # Go through each array!
            # I first!
            feature_index_I = reduction.get_features(X_train, y_M_train, model)
            model.fit(X_train[:,feature_index_I], y_M_train)

            y_pred = model.predict(X_cv[:,feature_index_I])
            assert(y_pred.shape[0] == test_index.shape[0])
            f1_value = f1_score(y_pred, y_I_cv)
            f1_I.append(f1_value)

            # H second!
            feature_index_H = reduction.get_features(X_train, y_M_train, model)
            model.fit(X_train[:,feature_index_H], y_M_train)

            y_pred = model.predict(X_cv[:,feature_index_H])
            assert(y_pred.shape[0] == test_index.shape[0])
            f1_value = f1_score(y_pred, y_H_cv)
            f1_H.append(f1_value)

            # M third!
            feature_index_M = reduction.get_features(X_train, y_M_train, model)
            model.fit(X_train[:,feature_index_M], y_M_train)

            y_pred = model.predict(X_cv[:,feature_index_M])
            assert(y_pred.shape[0] == test_index.shape[0])
            f1_value = f1_score(y_pred, y_M_cv)
            f1_M.append(f1_value)

        if verbose:
            print("Predicted average f1 scores are: ")
            print("F1 score for I: ", np.mean(f1_I))
            print("F1 score for M: ", np.mean(f1_M))
            print("F1 score for H: ", np.mean(f1_H))
            print("Feature indecies: ", feature_index_M)

        return np.mean(f1_H), np.mean(f1_I), np.mean(f1_M)

        #     # classifier with ensemble
        #     # ATTENTION: unbalanced data
        #     # adaboost for each of the problem
        #     # reweighting (later)
        #     # can play with K
        #     clf.fit(X_M_train, y_M_train)
        #     score = f1(X_M_validation, y_M_validation)
        #     print("Score " + score)

    def batch_run_models(self):

        all_reductions = [
            ("None", Identity()),
            ("RFE", RFEReduction())
        ]

        all_models = [
        #    ("Random Model: ", RandomModel()),
            #("tpot Model: ", TPot()),
            ("Adaboost", AdaBoostClassifier(n_estimators=10, learning_rate=1.))
        ]

        for model in all_models:
            for red in all_reductions:
                f1_H, f1_I, f1_M = runner.run(red[1], model[1])
                print(red[0] + ":" + model[0] + " has f1_H: " + str(f1_H) + ", f1_I: " + str(f1_I) + ", f1_M :" + str(f1_M))

    def meta_runners(self):
        # Call meta-algorithms here
        tpot_instance = TPot()
        tpot_instance.fit(self.X_train, self.y_H, "MODEL_y_H")
        tpot_instance.fit(self.X_train, self.y_I, "MODEL_y_I")
        tpot_instance.fit(self.X_train, self.y_M, "MODEL_y_M")

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
    runner.batch_run_models()
#    runner.run(rndModel)
#    runner.meta_runners()