import numpy as np
a = np.empty([2,2,2], dtype = np.float32)
b = np.empty([2,2,2], dtype = np.float32)

print(a)
#print(b)
c = a + b
#print(c)

for i in range(1, 10):
    print(i)
