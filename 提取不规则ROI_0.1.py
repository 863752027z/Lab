import cv2
import numpy as np

def extractROI(roi_corners, img):
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([roi_corners], dtype=np.int32)
    channel_count = img.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

path = 'E:/1.jpg'
image = cv2.imread(path)
roi_corners = [(10,10), (50,20), (90,60), (130,120)]
roi = extractROI(roi_corners, image)
cv2.imshow('1', roi)
cv2.waitKey(0)

