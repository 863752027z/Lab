import numpy as np
import cv2 as cv

img = np.zeros((320, 320, 3), np.uint8)  # 生成一个空灰度图像

# 起点和终点的坐标
ptStart = (60, 60)
ptEnd = (260, 260)
point_color = (0, 255, 0)  # BGR
thickness = 1
lineType = 4
cv.line(img, ptStart, ptEnd, point_color, thickness, lineType)

ptStart = (260, 60)
ptEnd = (60, 260)
point_color = (0, 0, 255)  # BGR
thickness = 1
lineType = 8
cv.line(img, ptStart, ptEnd, point_color, thickness, lineType)

cv.namedWindow("image")
cv.imshow('image', img)
cv.waitKey(10000)  # 显示 10000 ms 即 10s 后消失
cv.destroyAllWindows()
