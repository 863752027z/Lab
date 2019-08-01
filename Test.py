import numpy as np
import cv2

img = np.zeros((512,512,3), np.uint8)
cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.circle(img, (447,63), 63, (0, 0, 255), -1)
cv2.rectangle(img, (384,0), (510,128), (0,255,0), 3)
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
font = cv2.FONT_HERSHEY_SIMPLEX
#文字的字体类型，大小，颜色，粗细
cv2.putText(img, 'OpenCV', (10,500), font, 5, (255,255,255), 2)


cv2.imshow('img',img)
k = cv2.waitKey(0)&0xff
if k==27:
    cv2.destroyAllWindows()



[[[0. 2.]]

 [[1. 3.]]]
**********************
[[[2. 4.]]

 [[3. 5.]]]
