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

def addOpticalFlow(vectors):
    n = vectors.shape[0]
    result = [0,0]
    for i in range(n):
        result += vectors[i]
    return result



def imshow(name, img):
    cv2.imshow(name, img)
    k = cv2.waitKey(10)&0xff
    if k==27:
        cv2.destroyAllWindows()

def pre_process(initial_img, img, first, p0):
    img = face_cut(img)
    if first == True:
        p0 = face_detector(img)
        return p0, img
    else:
        p1 = face_detector(img)
        dst_point = p0
        src_point = p1
        temp_frame = AffineTransform(initial_img, img, src_point, dst_point)
        return temp_frame

def pre_solve1(video_path):
    frame = []
    step = 3
    count = 1
    cap = cv2.VideoCapture(video_path)
    total_frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #读step帧
    for i in range(step):
        temp_frame = cap.read()
        frame.append(temp_frame)





def pre_solve(video_path):
    count = 1
    cap = cv2.VideoCapture(video_path)
    total_frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame = []
    p0 = None
    dst_point = None
    step = 3
    end = total_frame_number - step

    while count <= total_frame_number:
        ret, temp_frame = cap.read()
        #切脸
        temp_frame = face_cut(temp_frame)
        if count==1:
            frame.append(temp_frame)
            p0 = face_detector(temp_frame)
            dst_point = np.float32([p0[0][0], p0[30][0], p0[50][0]])
        if count!=1:
            #如果不是第一帧就进行放射变换
            p1 = face_detector(temp_frame)
            src_point = np.float32([p1[0][0], p1[30][0], p1[50][0]])
            #仿射变换
            temp_frame = AffineTransform(frame[0], temp_frame, src_point, dst_point)
            frame.append(temp_frame)
        imshow('f', frame[count-1])
        cv2.imwrite('E:/save_img/' + str(count) + '.jpg', frame[count-1])
        count += 1
        print(count)

    print(len(frame))
    return frame, p0

def multiOpticalFlowAdd(p0, frame):
    n = len(frame)
    optical_flow = np.empty([68, n-1, 2], dtype = np.float32)
    for i in range(n):
        return





video_path = 'E:/s15/15_0101disgustingteeth.avi'
#frame, p0 = pre_solve(video_path)

