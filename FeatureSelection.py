import pandas as pd
import numpy as np
import operator
import math

from Scores import accuracy_score
from Splits import k_fold_cross_validation


class ForwardSelection:
    ''' Feature Brute Force Search
    Takes Pandas DataFrame in fit() method'''
    def __init__(self, knn, n_iter):
        self.knn = knn
        self.n_iter = n_iter
        self.folds = 5

    def fit(self, X, y):
        # Feature Brute Force Search
        # returns selected features' indexes
        self.X, self.y = X, y
        iteration, best_accuracy = 0, 0
        used_features = []
        n_features = len(X.columns)
        unused_features = list(range(n_features))
        while iteration < self.n_iter:
            best_unused_feature = None
            for feature in unused_features:
                features = used_features + [feature]
                accuracy = self.eval(features)
                print("With features: {}, acc = {}".format(features, accuracy))
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    best_unused_feature = feature
            if best_unused_feature is not None:
                used_features.append(best_unused_feature)
                unused_features.remove(best_unused_feature)
            else:
                break
            iteration += 1
        return self.feature_columns(used_features)

    def eval(self, features):
        features = self.X.iloc[:, features].values
        labels = self.y.values
        accuracies = 0
        for i in range(self.folds):
            X_train, y_train, X_test, y_test = k_fold_cross_validation(features, labels, i)
            self.knn.fit(X_train, y_train)
            y_pred = self.knn.predict(X_test)
            accuracies += accuracy_score(y_test, y_pred)
        return accuracies/self.folds

    def feature_columns(self, features_idxs):
        features = []
        for idx in features_idxs:
            column = self.X.columns[idx]
            features.append(column)
        return features


class VarianceThreshold:
    ''' Class that drops features with
    variance lower than specified constant
    '''
    def __init__(self, constant):
        self.constant = constant

    def fit(self, X):
        self.X = X
        self.variances = []
        for col in self.X.columns:
            column_values = np.array(self.X[col], dtype=str)
            self.variances.append(self.variance(column_values))
        return self

    def transform(self, X):
        features_to_drop = []
        for idx, variance in enumerate(self.variances):
            if variance < self.constant:
                feature = X.columns[idx]
                features_to_drop.append(feature)
        for feature in features_to_drop:
            print('Dropping "{}" because of small variance'.format(feature))
            X = X.drop(feature, axis=1)
        return X

    @staticmethod
    def variance(array):
        ''' This is an implementation of
        variance, same results can be produced
        with numpy function "np.var" '''
        item_to_prob = {}
        n_items = len(array)
        for item in array:
            if item_to_prob.get(item) is not None:
                item_to_prob[item] += 1
            else:
                item_to_prob[item] = 1
        squared_values_sum = 0
        values_sum = 0
        for pair in item_to_prob.items():
            value = pair[0]
            prob = pair[1]/n_items
            squared_values_sum += (float(value) ** 2) * prob
            values_sum += float(value) * prob
        values_sum_squared = values_sum ** 2
        variance = squared_values_sum - values_sum_squared
        return variance


class LinearlyIndependent:
    ''' Class that selects
    linearly independent features
    using Pearson Coeffiecient '''
    def __init__(self):
        pass

    def transform(self, X):
        ''' Algorithm description:
        Feature Brute Force Search
        Based on LinearIndependence '''
        n_features = len(X.columns)
        features_to_drop = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                a = X.iloc[:, [i]].values
                b = X.iloc[:, [j]].values
                if not self.linearly_independent(a, b):
                    feature_to_drop = X.columns[j]
                    features_to_drop.append(feature_to_drop)
        features_to_drop = np.unique(features_to_drop)
        for feature in features_to_drop:
            print('Dropping "{}" because of linear dependency'.format(feature))
            X = X.drop(feature, axis=1)
        return X

    @staticmethod
    def linearly_independent(a, b):
        a = a.astype('float32').flatten()
        b = b.astype('float32').flatten()
        # NumPy function was used instead of the below one
        # pearsons = LinearlyIndependent.pearsons(a, b)
        # because the numpy function is faster than mine
        pearsons = np.corrcoef(a,b)[0, 1]
        if abs(np.corrcoef(a,b)[0, 1]) >= 0.5:
            return False
        else:
            return True

    @staticmethod
    def mean(x):
        return sum(x)/len(x)

    @staticmethod
    def covariance(x,y):
        calc = []
        for i in range(len(x)):
            xi = x[i] - LinearlyIndependent.mean(x)
            yi = y[i] - LinearlyIndependent.mean(y)
            calc.append(xi * yi)
        return sum(calc)/(len(x))

    @staticmethod
    def stand_dev(x):
        variance = VarianceThreshold.variance(x)
        return math.sqrt(variance)

    @staticmethod
    def pearsons(x,y):
        cov = LinearlyIndependent.covariance(x,y)
        return cov / (LinearlyIndependent.stand_dev(x) * LinearlyIndependent.stand_dev(y))
