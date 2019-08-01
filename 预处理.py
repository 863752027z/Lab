import cv2
import numpy as np
import os

def get_path(remain_dir):
    path1 = 'E:/ZLW_generate_flow/' + remain_dir + '/'
    path2 = 'E:/ZLW_generate_flow_same/' + remain_dir + '/'
    path_list1 = []
    path_list2 = []
    #counnt=1表示进入到第一个目录下
    count = 1
    for root, dirs, files in os.walk(path1):
        if count == 1:
            #print(root)
            print(dirs)
            #print(files)
            for i in range(len(dirs)):
                temp_path = path1 + dirs[i]
                path_list1.append(temp_path)
                temp_path = path2 + dirs[i]
                path_list2.append(temp_path)
            break
        count+=1
    return path_list1, path_list2

def same_size(size, path1, path2):
    print('处理'+ path1)
    l= 0
    #处理path1, 写入path2
    for root, dirs, files in os.walk(path1):
        l = len(files)
        break
    for i in range(l):
        #当前帧数
        count = i+1
        temp_path = path1 + '/' + str(count) + '.jpg'
        frame = cv2.imread(temp_path)
        print('读取' + temp_path)
        frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_NEAREST)
        temp_path = path2 + '/' + str(count) + '.jpg'
        print('写入'+temp_path)
        cv2.imwrite(temp_path, frame)



size = [300,300]
remain_dir = ['s15']
for i in range(len(remain_dir)):
    path1, path2 = get_path(remain_dir[i])
    print(path1)
    print(path2)
    base_path = 'E:/ZLW_generate_flow_same/' + remain_dir[i]
    if not os.path.exists(base_path):
        os.mkdir(base_path)
        print('创建' + base_path)
    for i in range(len(path1)):
        if not os.path.exists(path2[i]):
            os.mkdir(path2[i])
            print('创建'+path2[i])
            same_size(size,path1[i], path2[i])
