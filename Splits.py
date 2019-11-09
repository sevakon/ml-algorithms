import numpy as np

def train_test_split(df, test_size=0.5):
    rows, columns = df.shape
    msk = np.random.rand(rows) < (1 - test_size)
    train = df[msk]
    test = df[~msk]
    return train, test

def k_fold_cross_validation(X, y, fold, n_folds=5):
    length = len(y)
    size = length/n_folds
    start = int(fold * size)
    end = int(start + size)
    X_test = X[start:end]
    y_test = y[start:end]
    X_train = np.concatenate((X[0:start], X[end:length]), axis=0)
    y_train = np.concatenate((y[0:start], y[end:length]), axis=0)
    return X_train, y_train, X_test, y_test
