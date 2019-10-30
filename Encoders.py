import pandas as pd
import numpy as np
import operator
import math

class CatEncoder:
    ''' Class that encodes categorial
    variables into positive numerical values,
    assuming that all rows values are categories
    '''
    def __init__(self):
        pass

    def fit(self, df):
        self.feature_values = {}
        for col in df.columns:
            column_values = np.array(df[col], dtype=str)
            self.feature_values[col] = sorted(np.unique(column_values))
        return self

    def transform(self, df):
        rows, columns = df.shape
        array = np.empty(df.shape)
        for j, col in enumerate(df.columns):
            for row in range(rows):
                value = str(df[col][row])
                index = self.feature_values[col].index(value)
                array[row, j] = index
        return array


class OneHotEncoder:
    ''' Class that encodes positive
    numerical values into one-hot vectors
    assuming that all values are numerical
    '''
    def __init__(self):
        pass

    def fit(self, array):
        self.unique_features = np.amax(array, axis=0)
        length = np.sum(self.unique_features) + len(self.unique_features)
        self.size = int(length)
        return self

    def transform(self, array):
        rows, columns = array.shape
        result = np.zeros([rows, self.size])
        for row in range(rows):
            for j, feature in enumerate(self.unique_features):
                col = int((feature + 1) * j + array[row, j])
                result[row, col] = 1
        return result
