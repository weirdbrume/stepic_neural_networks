from urllib.request import urlopen
import numpy as np

filename = input()
f = urlopen(filename)
data = np.loadtxt(f, skiprows=1, delimiter=',')

Y = data[:, 0].reshape(data.shape[0], 1)
B0 = np.ones_like(Y)
X = np.hstack((B0, data[:, range(1, data.shape[-1])]))
B_ = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))

print(' '.join(map(str, B_.flatten())))
