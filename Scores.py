import numpy as np

def accuracy_score(y_true, y_pred):
    ''' Predicted right to size ratio '''
    size = len(y_true)
    sum = 0
    for i in range(size):
        if y_true[i] == y_pred[i]:
            sum += 1
    return sum/size

def rmse(y_true, y_pred):
    ''' Root Mean Square Error '''
    size = len(y_true)
    square_sum = 0
    for i in range(size):
        square_sum += (y_true - y_pred) ** 2
    mean_square_sum = square_sum/size
    root_mean_square_size = sqrt(mean_square_sum)
    return root_mean_square_size
