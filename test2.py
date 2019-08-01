import numpy as np
import cv2

point1 = np.empty([0,1,2], dtype=np.float32)
point2 = np.empty([0,1,2], dtype=np.float32)


for i in range(2):
    pos = (i, i+2)
    temp_point = np.empty([1,1,2], dtype=np.float32)
    temp_point[0,0] = pos
    point1 = np.concatenate([point1, temp_point])
    pos = (i+2, i+4)
    temp_point = np.empty([1,1,2], dtype=np.float32)
    temp_point[0,0] = pos
    point2 = np.concatenate([point2, temp_point])

print(list(zip(point1, point2)))

for i,p in enumerate(point1):
    print(i,p)
""""
for i, (new, old) in enumerate(zip(point1, point2)):
    print(new)
"""

"""
for i, (new, old) in enumerate(zip(point1, point2)):
    c,d = old.ravel()
    pos = (c,d)
    print(pos)
"""