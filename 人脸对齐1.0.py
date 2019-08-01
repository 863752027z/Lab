import numpy as np
import itertools
import cv2
import os
import dlib

detector_face_cut = cv2.CascadeClassifier('E:/data/haarcascade_frontalface_default.xml')
def face_cut(img,count):
    original_frame = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector_face_cut.detectMultiScale(gray, 1.3, 5)
    while(faces.shape[1] == 0):
        print('未检测到人脸')
        print(faces)
        faces = detector_face_cut.detectMultiScale(gray, 1.1, 5)
    print('检测到人脸')
    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]
    img1 = img[y:y + h-1, x:x + w-1]
    img1 = cv2.resize(img1,(300,300))
    return img1

#original_frame是为了face_cut失败的时候重新face_cut
#读磁盘的速度太慢
detector_face_dector = dlib.get_frontal_face_detector()
predictor_face_dector = dlib.shape_predictor('E:/data/shape_predictor_68_face_landmarks.dat')
def face_detector(frame):
    face_key_point = []
    # cv2读取图像
    # 取灰度
    img_gray = frame
    img = frame
    # 人脸数rects
    rects = detector_face_dector(img_gray, 0)
    face_key_point = np.empty([0,1,2], dtype=np.float32)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor_face_dector(img,rects[i]).parts()])
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

def AffineTransform(dst_img, img, src_point, dst_point):
    rows, cols, ch = dst_img.shape
    M = cv2.getAffineTransform(src_point, dst_point)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def C(n, m):
    a = 1
    b = 1
    c = 1
    temp = n-m
    for i in range(n):
        a *= n
        n -= 1
    for j in range(m):
        b *= m
        m -= 1
    for k in range(temp):
        c *= temp
        temp -= 1
    return a/(b*c)

def get_path(remain_dir):
    path1 = 'E:/CAS(ME)^2/rawvideo/' + remain_dir + '/'
    path2 = 'E:/ZLW_generate_face_better/' + remain_dir + '/'
    path_list1 = []
    path_list2 = []
    #counnt=1表示进入到第一个目录下
    count = 1
    for root, dirs, files in os.walk(path1):
        if count == 1:
            #print(root)
            #print(dirs)
            #print(files)
            for i in range(len(files)):
                temp_path = path1 + files[i]
                path_list1.append(temp_path)
                temp_path = path2 + files[i][:-4]
                path_list2.append(temp_path)
            break
        count+=1
    return path_list1, path_list2

def getFeaturePoint_M(face_key_point, feature_point, one):
        M1 = np.empty([0,3], dtype=np.float32)
        for i in range(len(feature_point)):
            index = feature_point[i]
            temp_point = np.empty([1,3], np.float32)
            temp_point[0][0] = face_key_point[index][0][0]
            temp_point[0][1] = face_key_point[index][0][1]
            temp_point[0][2] = 1
            M1 = np.concatenate([M1, temp_point])
        M0 = M1[:, 0:2]
        M0 = np.transpose(M0)
        M1 = np.transpose(M1)
        if one:
            return M1
        else:
            return M0

def solve(path1, path2):
    cap = cv2.VideoCapture(path1)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #先用13个点
    feature_point = [1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 29]
    ROI = []
    first_frame = None
    #p0就是第一张图片的特征点
    p0 = None
    feature_point_M_one = None
    for i in range(total_frame):
        ret,temp_frame = cap.read()
        #保存刚读进来的一帧
        original_frame = temp_frame
        count = i+1
        temp_frame = face_cut(temp_frame, count)
        face_key_point = face_detector(temp_frame)
        while face_key_point.shape[0] != 68:
            temp_frame = face_cut(original_frame, count)
            face_key_point = face_detector(temp_frame)
        #处理第一帧的数据
        if count == 1:
            first_frame = temp_frame
            p0 = face_key_point
            feature_point_M_one = getFeaturePoint_M(p0, feature_point, None)
            print('处理' + str(count) + '.jpg')
            temp_path = path2 + '/' + str(count) + '.jpg'
            cv2.imwrite(temp_path, first_frame)
        else:
            #第二帧开始就要进行仿射变换矩阵的计算
            #左边1 3 5或者2 4 6，右边17 15 13或者16 14 12
            #鼻子29
            p1 = face_key_point
            min = 1000000
            minM = None
            feature_point_M_i = getFeaturePoint_M(p1, feature_point, True)
            for p in itertools.combinations(feature_point, 3):
                a = p[0]
                b = p[1]
                c = p[2]
                dst_point = np.float32([p0[a, 0], p0[b, 0], p0[c, 0]])
                src_point = np.float32([p1[a, 0], p1[b, 0], p1[c, 0]])
                temp_AffineM = cv2.getAffineTransform(src_point, dst_point)
                #计算2范数
                result = np.dot(temp_AffineM, feature_point_M_i)
                temp_M = result - feature_point_M_one
                norm2 = np.linalg.norm(temp_M, ord=2)
                if norm2 < min:
                    min = norm2
                    minM = temp_AffineM
            #算完所有的仿射变换矩阵，找到2范数最小的
            rows, cols, ch = first_frame.shape
            dst_img = cv2.warpAffine(temp_frame, minM, (cols, rows))
            print('处理' + str(count) + '.jpg')
            temp_path = path2 + '/' + str(count) + '.jpg'
            cv2.imwrite(temp_path, dst_img)

remain_dir = ['s15']
for i in range(len(remain_dir)):
    path1, path2 = get_path(remain_dir[i])
    print(path1)
    print(path2)
    base_path = 'E:/ZLW_generate_face_better/' + remain_dir[i]
    print(base_path)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    for i in range(len(path1)):
        if not os.path.exists(path2[i]):
            os.mkdir(path2[i])
            print('创建'+path2[i])
            solve(path1[i], path2[i])










