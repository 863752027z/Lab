import numpy as np
import itertools
import cv2
import os
import dlib
import math
import pandas as pd

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

def calcOpticalFlow(pre_frame, current_frame, p0):
    old_gray = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    lk_params = dict(winSize=(15,15),
                 maxLevel=2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_old = p0[st==1]
    good_new = p1[st==1]
    return p0,p1,st

def AffineTransform(dst_img, img, src_point, dst_point):
    rows, cols, ch = dst_img.shape
    M = cv2.getAffineTransform(src_point, dst_point)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

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

def get_path(remain_dir):
    path1 = 'E:/CAS(ME)^2/rawvideo/' + remain_dir + '/'
    path2 = 'E:/ZLW_mod_sub/' + remain_dir + '/'
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
        temp_point = (P[i][0], P[i][1])
        cv2.circle(img, temp_point, 1, color=(0, 255, 0))
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, str(i + 1), temp_point, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

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

def save_data_to_excel(data, path):
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter(path)
    data_df.to_excel(writer,'page_1', float_format='%.5f') # float_format 控制精度
    writer.save()

def read_img(path):
    ROI = []
    file_num, dir_num = get_file_number(path)
    file_num = file_num-1
    for i in range(file_num):
        filename = path + '/' + str(i+1) + '.jpg'
        temp_img = cv2.imread(filename)
        ROI.append(temp_img)
    return ROI

def mainProcess(path):
    ROI = read_img(path)
    frame = ROI
    print(len(ROI))
    count = 0
    nonzero = np.array(ROI[0].nonzero())
    P = np.empty([0, 1, 2], dtype=np.float32)
    mod = np.array([])
    angle = np.array([])
    good_new = None
    for j in range(nonzero.shape[1]):
        if (j+1)%3 == 0:
            pos = np.empty([1, 1, 2], dtype=np.float32)
            pos[0][0][0] = nonzero[1][j]
            pos[0][0][1] = nonzero[0][j]
            P = np.concatenate([P, pos])
    p0 = P
    '''
    print(p0.shape)
    draw_point(p0, ROI[0])
    cv2.imshow('1', ROI[0])
    cv2.waitKey(0)
    '''
    for i in range(len(frame) - 1):
        count += 1
        pre_frame = frame[i]
        current_frame = frame[i + 1]
        p0, p1, st = calcOpticalFlow(pre_frame, current_frame, p0)
        good_old = p0[st == 1]
        good_new = p1[st == 1]

        p0 = good_new.reshape(-1, 1, 2)
        p = []
        m = []
        a = []
        st = st.reshape(-1)
        if mod.any():
            mod = mod[st == 1]
        '''
        if pos.any():
            pos = pos[st == 1]
        '''
        if angle.any():
            angle = angle[st == 1]
        for j in range(good_old.shape[0]):
            v = [good_new[j][0] - good_old[j][0], good_new[j][1] - good_old[j][1]]
            temp_m = np.sqrt(np.dot(np.transpose(v), v))
            temp_a = math.atan2(good_old[j][1], good_old[j][0])
            m.append(temp_m)
            #p.append(good_old[j])
            a.append(temp_a)

        m = np.array(m)
        # p = np.array([p])
        a = np.array(a)

        if count == 1:
            mod = np.empty([good_old.shape[0], 1], dtype=np.float32)
            mod = np.column_stack((mod, m))
            mod = np.delete(mod, 0, axis=1)
            angle = np.empty([good_old.shape[0], 1], dtype=np.float32)
            angle = np.column_stack((angle, a))
            angle = np.delete(angle, 0, axis=1)
        else:
            if mod.any():
                mod = np.column_stack((mod, m))
            if angle.any():
                angle = np.column_stack((angle, a))

        print(i + 1)
        '''
        print(mod.shape[0])
        print(good_old)
        print(st)
        print('mod')
        print(mod)
        print('angle')
        print(angle)
        '''
        '''
        draw_point(good_old, frame[i])
        cv2.imshow('img', frame[i])
        cv2.waitKey(0)
        '''
        print(mod.shape[0])
    return mod, angle, good_new

def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] [lng,lat]
    #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1]==e_poi[1]: #排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1]>poi[1] and e_poi[1]>poi[1]: #线段在射线上边
        return False
    if s_poi[1]<poi[1] and e_poi[1]<poi[1]: #线段在射线下边
        return False
    if s_poi[1]==poi[1] and e_poi[1]>poi[1]: #交点为下端点，对应spoint
        return False
    if e_poi[1]==poi[1] and s_poi[1]>poi[1]: #交点为下端点，对应epoint
        return False
    if s_poi[0]<poi[0] and e_poi[0]<poi[0]: #线段在射线左边
        return False

    xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1]) #求交
    if xseg<poi[0]: #交点在射线起点的左侧
        return False
    return True  #排除上述情况之后

def isPoiWithinPoly(poi,poly):
    #输入：点，多边形三维数组
    #poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组

    #可以先判断点是否在外包矩形内
    #if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
    #但算最小外包矩形本身需要循环边，会造成开销，本处略去
    sinsc=0 #交点个数
    for epoly in poly: #循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
        for i in range(len(epoly)-1): #[0,len-1]
            s_poi=epoly[i]
            e_poi=epoly[i+1]
            if isRayIntersectsSegment(poi,s_poi,e_poi):
                sinsc+=1 #有交点就加1


area_list = [1, 2, 3, 4, 5, 6, 7, 8]
remain_dir = ['s15']
base_path = 'E:ZLW_data/'
for m in range(len(area_list)):
    for n in range(len(remain_dir)):
        path1 = base_path + str(area_list[m]) + '/' + remain_dir[n]
        for lists in os.listdir(path1):
            path2 = path1 + '/' + lists
            mod, angle, good_new = mainProcess(path2)
            mod_sub = np.empty([0, mod.shape[1]-1], dtype=np.float32)
            for i in range(mod.shape[0]):
                temp_sub = []
                for j in range(mod.shape[1]-1):
                    sub = mod[i][j+1] - mod[i][j]
                    temp_sub.append(sub)
                temp_sub = np.array([temp_sub])
                mod_sub = np.concatenate([mod_sub, temp_sub])
            save_path = base_path + 'mod_sub/'
            save_data_to_excel(mod_sub, save_path + str(area_list[m]) + '_' + remain_dir[n] + '_' + lists + '.xlsx')
            path2 = None
        path1 = None



