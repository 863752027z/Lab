import numpy as np
import cv2
import dlib
import os
import math

detector_face_cut = cv2.CascadeClassifier('E:/data/haarcascade_frontalface_default.xml')
def face_cut(img):
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

def get_file_number(path):
    file_num = 0
    dir_num = 0
    for lists in os.listdir(path):
        sub_path = os.path.join(path, lists)
        if os.path.isfile(sub_path):
            file_num += 1
        if os.path.isdir(sub_path):
            dir_num += 1
    return file_num, dir_num

def read_img(path):
    pos = (280, 220)
    w = 400
    h = 400
    ROI = []
    file_num, dir_num = get_file_number(path)
    for i in range(file_num):
        print(i)
        filename = path + '/' + str(i) + '.jpg'
        temp_img = cv2.imread(filename)
        ROI.append(temp_img)
        #cv2.imshow('img', temp_img)
        #cv2.waitKey(20)
    return ROI

def read_img(path, start):
    pos = (240, 230)
    w = 400
    h = 400
    ROI = []
    file_num, dir_num = get_file_number(path)
    for i in range(file_num):
        print(i+start)
        filename = path + '/' + str(i+start) + '.jpg'
        temp_img = cv2.imread(filename)
        print('读入' + filename)
        ROI.append(temp_img)
    return ROI

def viz_flow(flow):
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

path = 'D:/SAMM/006/1_part1'
save_path = 'D:/ZLW_data/SAMM/flow_part1_detect/'

start = 0
frame_list = read_img(path, start)
for i in range(len(frame_list)-1):
    print('当前帧数:' + str(i+start))
    pre_gray = cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame_list[i + 1], cv2.COLOR_BGR2GRAY)
    temp_flow = cv2.calcOpticalFlowFarneback(pre_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    bgr = viz_flow(temp_flow)
    cv2.imwrite(save_path + str(i+start) + '.jpg', bgr)
    print('写入:' + save_path + str(i+start) + '.jpg')



'''
path1 = 'D:/ZLW_data/SAMM/face_part1/'
path2 = 'D:/ZLW_data/SAMM/face_part2/'
'''
