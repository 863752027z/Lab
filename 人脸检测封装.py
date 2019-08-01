import cv2
import numpy as np



video_path = 'E:/s15/15_0101disgustingteeth.avi'
cap = cv2.VideoCapture(video_path)
ret, img = cap.read()
img1 = face_cut(img)
cv2.imshow("img1", img1)
cv2.waitKey(0)