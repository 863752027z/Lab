import numpy as np
import cv2
import dlib

def AffineTransform(dst_img, img, src_point, dst_point):
    rows, cols, ch = dst_img.shape
    M = cv2.getAffineTransform(src_point, dst_point)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

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

def calcOpticalFlow(old_gray, frame_gray, p0):
    lk_params = dict(winSize=(15,15),
                 maxLevel=2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_old = p0[st==1]
    good_new = p1[st==1]
    return good_old, good_new


video_path = 'E:/s15/15_0101disgustingteeth.avi'
color = np.random.randint(0, 255, (100, 3))

cap = cv2.VideoCapture(video_path)
ret, old_frame = cap.read()
old_frame = face_cut(old_frame)
initial_frame = old_frame
dst_img = old_frame

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = face_detector(old_frame)
mask = np.zeros_like(old_frame)
dst_point = np.float32([p0[0][0], p0[30][0], p0[50][0]])

"""
while(True):
    ret, frame = cap.read()
    frame = face_cut(frame)
    #第二帧关键点检测之后，仿射变换
    p1 = face_detector(frame)
    #特征点的数目
    n = p1.shape[0]
    src_point = np.float32([p1[0][0], p1[10][0], p1[20][0]])
    #调试
    print(src_point)
    print(dst_point)
    #调试

    frame = AffineTransform(initial_frame, frame, src_point, dst_point)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    good_old, good_new = calcOpticalFlow(old_gray, frame_gray, p0)

    #画图
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

    #更新
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
"""

while(True):
    ret, frame = cap.read()
    frame = face_cut(frame)
    # 第二帧关键点检测之后，仿射变换
    p1 = face_detector(frame)
    # 特征点的数目
    n = p1.shape[0]
    src_point = np.float32([p1[0][0], p1[30][0], p1[50][0]])
    # 调试
    print(src_point)
    print(dst_point)
    # 调试

    frame = AffineTransform(initial_frame, frame, src_point, dst_point)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    good_old, good_new = calcOpticalFlow(old_gray, frame_gray, p0)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(150)&0xff
    if k == 27:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)


