import cv2
import numpy as np


file_path = 'E:/reba.jpg'
img = cv2.imread(file_path)
rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('image', dst)
k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('E:/Rachel1.jpg', dst)
    cv2.destroyAllWindows()