import numpy as np
import itertools
import cv2
import os
import dlib

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

def get_path(remain_dir):
    path1 = 'E:/CAS(ME)^2/rawvideo/' + remain_dir + '/'
    path2 = 'E:/ZLW_generate_ROI_flow_+30/' + remain_dir + '/'
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
    temp = []
    for i in range(face_key_point.shape[0]):
        a = face_key_point[i][0][0]
        b = face_key_point[i][0][1]
        tp = (a, b)
        temp.append(tp)
    face_key_point = temp
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
    P3 = face_key_point[40-1]
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

def solve(first_frame, currrent_frame, FIRST, p0, feature_point_M_one):
    feature_point = [1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 29]
    #保存刚读进来的一帧
    original_frame = currrent_frame
    temp_frame = face_cut(original_frame)
    face_key_point = face_detector(temp_frame)
    while face_key_point.shape[0] != 68:
        temp_frame = face_cut(original_frame)
        face_key_point = face_detector(temp_frame)
    if FIRST:
        p0 = face_key_point
        feature_point_M_one = getFeaturePoint_M(p0, feature_point, False)
        return p0, temp_frame, feature_point_M_one
    else:
        #第二帧开始就要进行仿射变换矩阵的计算
        #左边1 3 5或者2 4 6，右边17 15 13或者16 14 12
        #鼻子29
        p1 = face_key_point
        min = 1000000
        minM = None
        #该点集矩阵需要进行仿射变换，所以需要True
        feature_point_M_i = getFeaturePoint_M(p1, feature_point, True)
        for p in itertools.combinations(feature_point, 3):
            a = p[0]-1
            b = p[1]-1
            c = p[2]-1
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
        return dst_img

def generate_flow(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    p0 = None
    first_frame = None
    feature_point_M_one = None
    step = 30
    frame = []
    ROI_frame = []
    ROI_flow = []
    add_ROI_flow = []
    area = None
    list = [5, 6]
    total_flow_number = total_frame - 1
    add_flow_number = total_flow_number - step + 1
    count = 1
    for i in range(1, add_flow_number+1):
        if i == 1:
            #计算第一个光流
            #先读第一帧
            ret, temp_frame = cap.read()
            p0, first_frame, feature_point_M_one = solve(None, temp_frame, True, None, None)
            frame.append(first_frame)
            area = set_area(p0)
            first_ROI_frame = extractROI_chg(area, [5, 6], first_frame)
            ROI_frame.append(first_ROI_frame)
            #再读step个帧
            for j in range(step):
                ret, temp_frame = cap.read()
                temp_frame = solve(first_frame, temp_frame, False, p0, feature_point_M_one)
                frame.append(temp_frame)

                temp_ROI = extractROI_chg(area, [5, 6], temp_frame)
                ROI_frame.append(temp_ROI)
            #计算稠密光流
            for j in range(step):
                pre_gray_ROI = cv2.cvtColor(ROI_frame[j], cv2.COLOR_BGR2GRAY)
                gray_ROI = cv2.cvtColor(ROI_frame[j+1], cv2.COLOR_BGR2GRAY)
                temp_flow_ROI = cv2.calcOpticalFlowFarneback(pre_gray_ROI, gray_ROI, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                ROI_flow.append(temp_flow_ROI)
            #累加光流
            temp_add_flow_ROI = np.zeros_like(ROI_flow[0])
            for k in range(step):
                temp_add_flow_ROI += ROI_flow[k]
            add_ROI_flow.append(temp_add_flow_ROI)
            bgr = viz_flow(add_ROI_flow[0])
            #cv2.imshow('img', bgr)
            #cv2.waitKey(1)
            cv2.imwrite(save_path + '/' + str(i) + '.jpg', bgr)
        else:
            #计算第二个以后的光流累加
            ret, temp_frame = cap.read()
            temp_frame = solve(first_frame, temp_frame, False, p0, feature_point_M_one)
            pre_gray_ROI = cv2.cvtColor(ROI_frame[-1], cv2.COLOR_BGR2GRAY)
            frame.append(temp_frame)
            temp_ROI = extractROI_chg(area, [5,6], temp_frame)
            ROI_frame.append(temp_ROI)
            gray_ROI = cv2.cvtColor(ROI_frame[-1], cv2.COLOR_BGR2GRAY)
            temp_flow_ROI = cv2.calcOpticalFlowFarneback(pre_gray_ROI, gray_ROI, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            ROI_flow.append(temp_flow_ROI)
            temp_add_flow_ROI = np.zeros_like(ROI_flow[0])
            #打印出累加的光流
            for m in range(step):
                temp_add_flow_ROI += ROI_flow[i-1+m]
                print('累加'+ str(i-1+m))
            print('end')
            #打印累加的光流
            add_ROI_flow.append(temp_add_flow_ROI)
            bgr = viz_flow(temp_add_flow_ROI)
            image = np.concatenate([temp_ROI, temp_frame, bgr], axis=1)
            cv2.imwrite(save_path + '/' + str(i) + '.jpg', bgr)

remain_dir = ['s15']
for i in range(len(remain_dir)):
    path1, path2 = get_path(remain_dir[i])
    print(path1)
    print(path2)
    base_path = 'E:/ZLW_generate_ROI_flow_+30/' + remain_dir[i]
    print(base_path)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    for i in range(len(path1)):
        if not os.path.exists(path2[i]):
            os.mkdir(path2[i])
            print('创建'+path2[i])
            generate_flow(path1[i], path2[i])










