import cv2
import numpy as np

base_path = 'E:/concat/'
path1 = base_path + '1.jpg'
path2 = base_path + '2.jpg'
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
image = np.concatenate([img1, img2], axis=1)
cv2.imshow('1', image)
cv2.waitKey(0)
