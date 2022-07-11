import numpy as np
import pandas as pd


def __varcharProcessing__(X, varchar_process="dummy_dropfirst"):
    dtypes = X.dtypes
    if varchar_process == "drop":
        X = X.drop(columns=dtypes[dtypes == np.object].index.tolist())
        print("Character Variables (Dropped):", dtypes[dtypes == np.object].index.tolist())
    elif varchar_process == "dummy":
        X = pd.get_dummies(X, drop_first=False)
        print("Character Variables (Dummies Generated):", dtypes[dtypes == np.object].index.tolist())
    elif varchar_process == "dummy_dropfirst":
        X = pd.get_dummies(X, drop_first=True)
        print("Character Variables (Dummies Generated, First Dummies Dropped):",
              dtypes[dtypes == np.object].index.tolist())
    else:
        X = pd.get_dummies(X, drop_first=True)
        print("Character Variables (Dummies Generated, First Dummies Dropped):",
              dtypes[dtypes == np.object].index.tolist())

    X["intercept"] = 1
    cols = X.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X = X[cols]

    return X