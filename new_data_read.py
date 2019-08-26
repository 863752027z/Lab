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
    pos = (280, 220)
    w = 400
    h = 400
    ROI = []
    file_num, dir_num = get_file_number(path)
    for i in range(file_num):
        print(i)
        filename = path + '/' + str(i) + '.jpg'
        temp_img = cv2.imread(filename)
        temp_img = temp_img[pos[1]+20:pos[1]+400-10, pos[0]+10:pos[0]+400-30]
        ROI.append(temp_img)
    return ROI

def read_img(path, start):
    pos = (280, 220)
    w = 400
    h = 400
    ROI = []
    file_num, dir_num = get_file_number(path)
    for i in range(file_num):
        print(i+start)
        filename = path + '/' + str(i+start) + '.jpg'
        temp_img = cv2.imread(filename)
        temp_img = temp_img[pos[1]+20:pos[1]+400-10, pos[0]+10:pos[0]+400-30]
        ROI.append(temp_img)
    return ROI

path = 'D:/SAMM/006/1_part2'
save_path = 'D:/ZLW_data/SAMM/face_part2/'
start = 6200
frame_list = read_img(path, start)
print(len(frame_list))
for i in range(len(frame_list)):
    save = save_path + str(i) + '.jpg'
    cv2.imwrite(save, frame_list[i])
    print('写入' + save)
    save = None
