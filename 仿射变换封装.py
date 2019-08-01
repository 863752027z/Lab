import cv2
import numpy as np
def face_cut(img):
    k = detector = cv2.CascadeClassifier('E:/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]
    img1 = img[y:y+h, x:x+w]
    return img1

def face_detector(frame):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('E:/shape_predictor_68_face_landmarks.dat')
    face_key_point = []
    # cv2读取图像
    # 取灰度
    img_gray = frame
    img = frame
    # 人脸数rects
    rects = detector(img_gray, 0)
    face_key_point = np.empty([0,1,2], dtype=np.float32)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            #print(idx,pos)
            temp_point = np.empty([1,1,2], dtype=np.float32)
            temp_point[0,0] = pos
            face_key_point = np.concatenate([face_key_point, temp_point])
            """
            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img, pos, 5, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)
            """
    return face_key_point
def AffineTransform(img, src_point, dst_point):
    rows, cols, ch = img.shape
    M = cv2.getAffineTransform(src_point, dst_point)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

video_path = 'E:/s15/15_0101disgustingteeth.avi'
cap = cv2.VideoCapture(video_path)
ret, img = cap.read()
img = face_cut(img)

ret, img1= cap.read()
img1 = face_cut(img1)
print(img1.shape)
pts2 = np.float32([[25, 94], [27, 130], [32, 163]])
pts1 = np.float32([[25, 94], [26, 130], [32, 163]])
dst = AffineTransform(img1, pts1, pts2)
cv2.imshow('image', dst)
k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('E:/rebagai.jpg', dst)
    cv2.destroyAllWindows()


"""
img = cv2.imread('E:/reba.jpg')
rows, cols, ch = img.shape

pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

dst = AffineTransform(img, pts1, pts2)

cv2.imshow('image', dst)
k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('E:/rebagai.jpg', dst)
    cv2.destroyAllWindows()
"""