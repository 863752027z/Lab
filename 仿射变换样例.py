import cv2
import numpy as np
"""
AffineMatrix = cv2.getAffineTransform(np.array(SrcPointsA),
                                      np.array(CanvasPointsA))
AffineImg = cv2.warpAffine(Img, AffineMatrix, (Img.shape[1], Img.shape[0]))
"""

img = cv2.imread('E:/reba.jpg')
rows, cols, ch = img.shape

pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[0, 10], [cols - 2, 1], [2, rows - 2]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('image', dst)
k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('Rachel1.jpg', dst)
    cv2.destroyAllWindows()