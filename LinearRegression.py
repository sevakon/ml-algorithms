import numpy as np

class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X, y):
        ''' Calculating coefficients
        intercept - b0, coef - b1...bn '''
        X, y, X_offset, y_offset, X_scale = \
        self.process(X.astype('float64'), y.astype('float64'))
        self.coef, residues, rank, singular = np.linalg.lstsq(X, y)
        self.coef = self.coef.T / X_scale
        self.intercept = y_offset - np.dot(X_offset, self.coef.T)
        return self

    @staticmethod
    def process(X, y):
        X_offset = np.average(X, axis=0)
        X -= X_offset
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        y = y.astype('float64')
        y_offset = np.average(y, axis=0)
        y -= y_offset
        return X, y, X_offset, y_offset, X_scale

    def predict(self, X):
        return np.dot(X, self.coef.T) + self.intercept
