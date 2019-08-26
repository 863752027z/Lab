import numpy as np

a = np.array([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]])
print(a.shape)
b = a.ravel()
print(b.shape)
print(a)
print(b)