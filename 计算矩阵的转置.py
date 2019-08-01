import numpy as np

def calcNorm2(m):
    mT = np.transpose(m)
    mTm = np.dot(mT, m)
    value, vector = np.linalg.eig(mTm)
    max_value = np.max(value)
    return max_value

m = np.arange(4).reshape(2,2)
max_value = calcNorm2(m)
print(max_value)



