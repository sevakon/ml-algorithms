import pandas as pd
import numpy as np
import operator
import math

# --------------------------- DISTANCE FUNCTIONS ---------------------------- #

def euclidean(x, y):
    ''' Euclidean Distance '''
    return np.sqrt(np.sum((x - y) ** 2))

def manhattan(x, y):
    ''' Manhattan Distance '''
    return np.sum(abs(x - y))

def cosine(x, y):
    ''' Cosine Distance '''
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

# ---------------------------- TRAINING FUNCTIONS --------------------------- #

def accuracy_score(y_true, y_pred):
    size = len(y_true)
    sum = 0
    for i in range(size):
        if y_true[i] == y_pred[i]:
            sum += 1
    return sum/size

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

# --------------------------- K-NEAREST NEIGHBORS --------------------------- #

class KNeighboursClassifier:
    ''' K-Nearest Neighbors '''
    def __init__(self, n_neighbours, metric):
        self.n_neighbours = n_neighbours
        self.metric = metric

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y
        self.size = len(X)

    def kneighbors(self, x):
        distances = []
        for i in range(self.size):
            distance = self.metric(self.x_train[i], x)
            distances += [(distance, i)]
        distances.sort(key=operator.itemgetter(0))
        neighbors = []
        for k in range(self.n_neighbours):
            neighbors += [distances[k][1]]
        return neighbors

    def vote(self, kneighbors):
        votes = {}
        for neighbor in kneighbors:
            vote = self.y_train[neighbor]
            if vote in votes:
                votes[vote] += 1
            else:
                votes[vote] = 1
        sorted_votes = sorted(votes.items(),
                              key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]

    def predict(self, X):
        y_pred = []
        for x in X:
            kneighbors = self.kneighbors(x)
            pred = self.vote(kneighbors)
            y_pred += [pred]
        return y_pred
