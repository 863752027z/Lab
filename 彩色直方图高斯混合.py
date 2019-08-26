import cv2
import numpy as np
import os
import pandas as pd
from sklearn.mixture import GaussianMixture

def calc_bgr_count(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    count_b = np.zeros(256, np.float)
    count_g = np.zeros(256, np.float)
    count_r = np.zeros(256, np.float)

    for i in range(height):
        for j in range(width):
            (b, g, r) = img[i, j]
            index_b = int(b)
            index_g = int(g)
            index_r = int(r)
            count_b[index_b] = count_b[index_b] + 1
            count_g[index_g] = count_g[index_g] + 1
            count_r[index_r] = count_r[index_r] + 1

    # 计算每一个通道的概率
    total = height * width
    count_b = count_b / total
    count_g = count_g / total
    count_r = count_r / total
    '''
    # 绘图
    x = np.linspace(0, 256, 256)

    y1 = count_b
    plt.figure()
    plt.bar( x, y1, 0.9, alpha = 1, color = 'b' )

    y2 = count_g
    plt.figure()
    plt.bar( x, y2, 0.9, alpha = 1, color = 'g' )

    y3 = count_r
    plt.figure()
    plt.bar( x, y3, 0.9, alpha = 1, color = 'r' )

    plt.show()
    '''
    return count_b, count_g, count_r

def bgr_count_form(count_b, count_g, count_r):
    count_b = np.array([np.transpose(count_b)], dtype=np.float32)
    count_g = np.array([np.transpose(count_g)], dtype=np.float32)
    count_r = np.array([np.transpose(count_r)], dtype=np.float32)
    m = np.concatenate([count_b, count_g])
    m = np.concatenate([m, count_r])
    m = m.ravel()
    return m

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

def read_img(path, start):
    ROI = []
    file_num, dir_num = get_file_number(path)
    #print('num:'+ str(file_num))
    for i in range(file_num):
        print(i+start)
        filename = path + '/' + str(i+start) + '.jpg'
        temp_img = cv2.imread(filename)
        ROI.append(temp_img)
    return ROI

def save_data_to_excel(data, path):
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter(path)
    data_df.to_excel(writer, 'page_1', float_format='%.5f') # float_format 控制精度
    writer.save()

data_path = 'D:/ZLW_data/SAMM/flow_cut_eye'
frame_list = read_img(data_path, 0)
data_set = np.empty([0, 256*3], dtype=np.float32)
for i in range(len(frame_list)):
    count_b, count_g, count_r = calc_bgr_count(frame_list[i])
    m = bgr_count_form(count_b, count_g, count_r)
    m = np.array([m])
    data_set = np.concatenate([data_set, m])
    print(data_set.shape)

gmm = GaussianMixture(n_components=2).fit(data_set)
labels = gmm.predict(data_set)
save_data_to_excel(labels, 'D:/guass/labels.xlsx')
frame_list0 = []
frame_list1 = []
for i in range(labels.shape[0]):
    if labels[i] == 0:
        print(str(i) + '是标签1')
        frame_list0.append(i)
for i in range(labels.shape[0]):
    if labels[i] == 1:
        print(str(i) + '是标签0')
        frame_list1.append(i)
frame_list0 = np.transpose(np.array([frame_list0]))
frame_list1 = np.transpose(np.array([frame_list1]))
save_data_to_excel(frame_list0, 'D:/guass/frame00.xlsx')
save_data_to_excel(frame_list1, 'D:/guass/frame01.xlsx')
