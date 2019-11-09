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
