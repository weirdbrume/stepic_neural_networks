import numpy as np


def matrix_from_input():
    matrix_shape = tuple(map(int, input().split()))
    matrix = np.fromiter(map(int, input().split()), np.int).reshape(matrix_shape)
    return matrix

X = matrix_from_input()
Y = matrix_from_input()

try:
    print(X.dot(Y.T))
except ValueError:
    print('matrix shapes do not match')
