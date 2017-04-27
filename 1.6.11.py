from urllib.request import urlopen
import numpy as np

filename = input()
f = urlopen(filename)
m = np.loadtxt(f, skiprows=1, delimiter=',')
print(m.mean(axis=0))
