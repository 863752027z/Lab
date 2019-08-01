import cv2
import numpy as np
import dlib
import os
import matplotlib.pyplot as plt

def AffineTransform(dst_img, img, src_point, dst_point):
    rows, cols, ch = dst_img.shape
    M = cv2.getAffineTransform(src_point, dst_point)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

#读磁盘的速度太慢
detector_face_cut = cv2.CascadeClassifier('E:/data/haarcascade_frontalface_default.xml')
def face_cut(img,count):
    original_frame = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector_face_cut.detectMultiScale(gray, 1.1, 5)
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
    img1 = cv2.resize(img1, (300, 300))
    return img1

#original_frame是为了face_cut失败的时候重新face_cut
#读磁盘的速度太慢
detector_face_dector = dlib.get_frontal_face_detector()
predictor_face_dector = dlib.shape_predictor('E:/shape_predictor_68_face_landmarks.dat')
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

def addOpticalFlow(vectors):
    n = vectors.shape[0]
    result = [0,0]
    for i in range(n):
        result += vectors[i]
    return result

def imshow(name, img):
    cv2.imshow(name, img)
    k = cv2.waitKey(0)&0xff
    return k


def pre_process(initial_img, img, first, p0, count):
    a = 1
    b = 17
    c = 28
    d = 40
    e = 46
    #保存原始图像
    original_img = img
    img = face_cut(original_img, count)

    face_key_point = face_detector(img)
    while face_key_point.shape[0] != 68:
        img = face_cut(original_img, count)
        face_key_point = face_detector(img)

    if first == True:
        p0 = face_key_point
        return p0, img
    else:
        p1 = face_key_point
        dst_point = np.float32([p0[a,0], p0[b,0], p0[c,0]])
        #print(dst_point)
        src_point = np.float32([p1[a,0], p1[b,0], p1[c,0]])
        temp_frame = AffineTransform(initial_img, img, src_point, dst_point)
        return temp_frame


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

def show_gray_hist(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.hist(img.ravel(), 256)
    plt.show()

def show_bgr_hist(img):
    color = ('b', 'g', 'r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(histr, color =col)
        plt.xlim([0,256])
    plt.show()

def get_path(remain_dir):
    path1 = 'E:/CAS(ME)^2/rawvideo/' + remain_dir + '/'
    path2 = 'E:/ZLW_generate_face_0.1/' + remain_dir + '/'
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

def debug_s19(video_path):
    initial_frame = None
    p0 = None
    frame = []
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('总帧数:' + str(total_frame))
    for i in range(total_frame):
        #读取一帧
        ret, temp_frame = cap.read()
        count = i + 1
        print('当前帧数:' + str(count))
        if count == 1:
            p0, initial_frame = pre_process(None, temp_frame, True, p0, count)
            frame.append(initial_frame)
            cv2.imwrite('E:/s19_debug1/s19/19_0402beatingpregnantwoman/' + str(count) + '.jpg', initial_frame)
        else:
                temp_frame = pre_process(initial_frame, temp_frame, False, p0, count)
                frame.append(temp_frame)
                if imshow('img', temp_frame) == 27:
                    cv2.destroyAllWindows()

def generate_face(path1, path2):
    print('处理'+ path1)
    video_path = path1
    initial_frame = None
    p0 = None
    frame = []
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('总帧数:' + str(total_frame))
    for i in range(total_frame):
        # 读取一帧
        ret, temp_frame = cap.read()
        count = i + 1
        print('当前帧数:' + str(count))
        if count == 1:
            p0, initial_frame = pre_process(None, temp_frame, True, p0, count)
            frame.append(initial_frame)
            cv2.imwrite(path2 + '/' + str(count) + '.jpg', initial_frame)
            print('生成' + path2 + '/' + str(count) + '.jpg')
        else:
            temp_frame = pre_process(initial_frame, temp_frame, False, p0, count)
            frame.append(temp_frame)
            cv2.imwrite(path2 + '/' + str(count) + '.jpg', temp_frame)
            print('生成' + path2 + '/' + str(count) + '.jpg')

remain_dir = ['s15']
for i in range(len(remain_dir)):
    path1, path2 = get_path(remain_dir[i])
    print(path1)
    print(path2)
    base_path = 'E:/ZLW_generate_face_0.1' + '/' + remain_dir[i]
    if not os.path.exists(base_path):
        os.mkdir(base_path)
        print('创建'+base_path)
    for i in range(len(path1)):
        if not os.path.exists(path2[i]):
            os.mkdir(path2[i])
            print('创建'+path2[i])
            generate_face(path1[i], path2[i])



