from sklearn.model_selection import KFold

from importer import Importer

class Runner():

    def __init__(self):
        print("Starting runer")
        self.imp = Importer()
        self.X_train, self.y_H, self.y_I, self.y_H

    def run(self):
        """
            Selects the best mode
        """
        print("Run!")

        kf = KFold(n_splits=10, shuffle=True)
        for train_index, validation_index in kf.split(self.X_train):
        #     X_M_train, X_M_validation = X_M[train_index], X_M[validation_index]
        #     y_M_train, y_M_validation = y_M[train_index], y_M[validation_index]
        #
        #     X_I_train, X_I_validation = X_I[train_index], X_I[validation_index]
        #     y_I_train, y_I_validation = y_I[train_index], y_I[validation_index]
        #
        #     X_H_train, X_H_validation = X_H[train_index], X_H[validation_index]
        #     y_H_train, y_H_validation = y_H[train_index], y_H[validation_index]
        #
        #     # classifier with ensemble
        #     # ATTENTION: unbalanced data
        #     # adaboost for each of the problem
        #     # reweighting (later)
        #     # can play with K
        #     clf.fit(X_M_train, y_M_train)
        #     score = f1(X_M_validation, y_M_validation)
        #     print("Score " + score)



if __name__ == "__main__":
    print("Starting runner!")
    imp = Importer()
    X_train, y_M, y_I, y_H = imp.get_X_y()
    print(X_train.head(5))
    print(y_M.head(5))
    print(y_I.head(5))
    print(y_H.head(5))

    runner = Runner()
    runner.run()