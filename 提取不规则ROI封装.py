import numpy as np
import cv2
import dlib

def face_detector(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('E:/shape_predictor_68_face_landmarks.dat')
    face_key_point = []
    # cv2读取图像
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            #print(idx,pos)
            face_key_point.append(pos)
    return face_key_point

def extractROI(roi_corners, image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([roi_corners], dtype=np.int32)
    channel_count = img.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def extractROI_chg(area,list, image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
    for i in range(len(list)):
        n = list[i]
        temp_corners = area[n]
        temp_corners = np.array([temp_corners], dtype=np.int32)
        cv2.fillPoly(mask, temp_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def draw_point(P, img):
    for i in range(len(P)):
        temp_point = P[i]
        cv2.circle(img, temp_point, 1, color=(0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i + 1), temp_point, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

def get_medin(face_key_point, n1, n2):
    index1 = n1-1
    index2 = n2-1
    x1 = face_key_point[index1][0]
    y1 = face_key_point[index1][1]
    x2 = face_key_point[index2][0]
    y2 = face_key_point[index2][1]
    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    pos1 = (x1, y1)
    pos2 = (x2, y2)
    pos = (x, y)
    return pos

def set_area(face_key_point):
    #左眉毛
    left_brow = np.empty([0, 2], dtype=np.float32)
    left_brow_point = [1, 18, 19, 20, 21, 22, 40, 39, 37, 2]
    left_brow = []
    for i in range(len(left_brow_point)):
        index = left_brow_point[i]-1
        temp_point = face_key_point[index]
        left_brow.append(temp_point)

    #右眉毛
    right_brow_point = [17, 27, 26, 25, 24, 23, 43, 44, 46, 16]
    right_brow = []
    for i in range(len(right_brow_point)):
        index = right_brow_point[i] - 1
        temp_point = face_key_point[index]
        right_brow.append(temp_point)

    #鼻子
    P1 = face_key_point[30-1]
    P2 = get_medin(face_key_point, 30, 14)
    P3 = get_medin(face_key_point, 31, 13)
    P4 = get_medin(face_key_point, 34, 51)
    P5 = get_medin(face_key_point, 5, 31)
    P6 = get_medin(face_key_point, 4, 30)
    nose = [P1, P2, P3, P4, P5, P6]

    #嘴巴
    P1 = face_key_point[34-1]
    P2 = get_medin(face_key_point, 34, 12)
    P3 = get_medin(face_key_point, 56, 11)
    P4 = get_medin(face_key_point, 9, 58)
    P5 = get_medin(face_key_point, 7, 60)
    P6 = get_medin(face_key_point, 6, 34)
    mouth = [P1, P2, P3, P4, P5, P6]

    #左眉毛1
    var = 20
    P1 = (face_key_point[20-1][0], face_key_point[25-1][1]-var)
    P2 = (face_key_point[22-1][0], face_key_point[22-1][1]-var)
    P3 = (face_key_point[40-1])
    P4 = face_key_point[39-1]
    P5 = face_key_point[37-1]
    P6 = face_key_point[1-1]
    P7 = (face_key_point[18-1][0], face_key_point[18-1][1]-var)
    left_brow1 = [P1, P2, P3, P4, P5, P6, P7]

    #右眉毛1
    var = 20
    P1 = (face_key_point[25-1][0], face_key_point[25-1][1]-var)
    P2 = (face_key_point[27-1][0], face_key_point[27-1][1]-var)
    P3 = face_key_point[17-1]
    P4 = face_key_point[46-1]
    P5 = face_key_point[44-1]
    P6 = face_key_point[43-1]
    P7 = (face_key_point[23-1][0], face_key_point[23-1][1]-var)
    right_brow1 = [P1, P2, P3, P4, P5, P6, P7]

    #左眼下方
    P1 = face_key_point[40-1]
    P2 = face_key_point[29-1]
    P3 = face_key_point[30-1]
    P4 = face_key_point[4-1]
    P5 = face_key_point[3-1]
    P6 = face_key_point[42-1]
    P7 = face_key_point[41-1]
    left_eye_below = [P1, P2, P3, P4, P5, P6, P7]

    #右眼下方
    P1 = face_key_point[43-1]
    P2 = face_key_point[48-1]
    P3 = face_key_point[47-1]
    P4 = face_key_point[15-1]
    P5 = face_key_point[14-1]
    P6 = face_key_point[30-1]
    P7 = face_key_point[29-1]
    right_eye_below = [P1, P2, P3, P4, P5, P6, P7]
    area = {1:left_brow, 2:right_brow, 3:nose, 4:mouth, 5:left_brow1, 6:right_brow1, 7:left_eye_below, 8:right_eye_below}
    return area

def set_point(face_key_point, img):
    for i in range(len(face_key_point)):
        temp_point = face_key_point[i]
        cv2.circle(img, temp_point, 1, color=(0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i+1), temp_point, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite('E:/feature_point.jpg', img)

def show_area(area, list,img):
    print(list)
    masked_iamge = extractROI_chg(area, list, img)
    cv2.imwrite('E:/ZLW_area/' + 'area' + '.jpg', masked_iamge)

img = cv2.imread('E:/face_point.jpg')
face_key_point = face_detector(img)
area = set_area(face_key_point)
#show_area(area, face_key_point)
list = []
for i in range(len(area)):
    if i >= 2:
        list.append(i+1)
list = [5, 6]
show_area(area, list, img)

'''
path = 'E:/1.jpg'
image = cv2.imread(path)
roi_corners = [(10,10), (50,20), (90,60), (130,120)]
roi = extractROI(roi_corners, image)
cv2.imshow('1', roi)
cv2.waitKey(0)
'''

'''
# mask defaulting to black for 3-channel and transparent for 4-channel
# (of course replace corners with yours)
mask = np.zeros(image.shape, dtype=np.uint8)
roi_corners = np.array([[(10,10), (50,20), (90,60), (130,120)]], dtype=np.int32)
# fill the ROI so it doesn't get wiped out when the mask is applied
channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
ignore_mask_color = (255,)*channel_count

cv2.fillPoly(mask, roi_corners, ignore_mask_color)
# from Masterfool: use cv2.fillConvexPoly if you know it's convex

# apply the mask
masked_image = cv2.bitwise_and(image, mask)
'''