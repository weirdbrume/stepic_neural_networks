import numpy as np

mat = np.eye(3, 4, 1) + np.eye(3, 4) * 2
mat = np.array(mat)
print(mat.reshape(12, 1))
