import numpy as np

X = np.array([[1, 60], [1, 50], [1, 75]])
Y = np.array([[10], [7], [12]])

print(np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y)))
