import numpy as np
import cv2
import dlib
import os
import math

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
    ROI = []
    file_num, dir_num = get_file_number(path)
    file_num = file_num-1
    for i in range(file_num):
        filename = path + '/' + str(i+1) + '.jpg'
        temp_img = cv2.imread(filename)
        ROI.append(temp_img)
    return ROI

def calcOpticalFlow(pre_frame, current_frame, p0):
    old_gray = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    lk_params = dict(winSize=(15,15),
                 maxLevel=2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_old = p0[st==1]
    good_new = p1[st==1]
    return p0, p1, st

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
        if (j+1) % 3 == 0:
            pos = np.empty([1, 1, 2], dtype=np.float32)
            pos[0][0][0] = nonzero[1][j]
            pos[0][0][1] = nonzero[0][j]
            P = np.concatenate([P, pos])
    p0 = P

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

def read_img(path):
    w = 400
    frame_list = []
    cap = cv2.VideoCapture(path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('帧数:' + str(total_frame))
    total_frame -= 2
    for i in range(45):
        print(i+1)
        ret, temp_frame = cap.read()
        temp_frame = temp_frame[200:720-270, w+170:1280-w-50]
        frame_list.append(temp_frame)
        cv2.imshow('img', temp_frame)
        cv2.waitKey(100)
    return frame_list

path = 'D:/Video/my_simile.mp4'
frame_list = read_img(path)
