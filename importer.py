import pandas as pd
import numpy as np

DEV = True
DEV_SIZE = 5

class Importer:

    def __init__(self):
        print("Importing...")
        self.data_basepath = "./data/"
        self.test_filepath = self.data_basepath + "exprs_test.csv"
        self.train_filepath = self.data_basepath + "exprs_train.csv"
        self.train_labels_filepath = self.data_basepath + "pathology_train.csv"

        self.import_raw_train()

        print("Import done!")


    def import_raw_train(self, verbose=False):
        self.X_train = pd.read_csv(self.train_filepath, nrows=DEV_SIZE) if DEV else pd.read_csv(self.train_filepath)
        self.X_train = self.X_train.sort_values("sample_name")

        self.y_train = pd.read_csv(self.train_labels_filepath, nrows=DEV_SIZE) if DEV else pd.read_csv(self.train_labels_filepath)
        self.y_train = self.y_train.sort_values("sample_name")


        if verbose:
            print(self.X_train.head(5))
            print(self.y_train.head(5))

    def get_X_y(self):
        return np.asarray(self.X_train.loc[:, self.X_train.columns != "sample_name"]), \
               np.asarray(self.y_train.loc[:, self.y_train.columns == "Microgranuloma"]).reshape(-1), \
               np.asarray(self.y_train.loc[:, self.y_train.columns == "Increasedmitosis"]).reshape(-1),\
               np.asarray(self.y_train.loc[:, self.y_train.columns == "Hypertrophy"]).reshape(-1)

if __name__ == "__main__":
    print("Starting run!")