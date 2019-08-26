import cv2
import os

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

def convert_video2img(path, save_path, start):
    w = 640
    h = 480
    cap = cv2.VideoCapture(path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frame)
    frame_list = []
    for i in range(total_frame):
        filename = path + '/' + str(i+start) + '.jpg'
        ret, temp_img = cap.read()
        temp_img = temp_img[120:h-110, 210+3:w-175-2]
        frame_list.append(temp_img)
        save = save_path + str(i+1) + '.jpg'

        if (i+1) > 1200:
            print('写入' + save)
            cv2.imshow('img', temp_img)
            cv2.waitKey(30)
        #cv2.imwrite(save, temp_img)
        save = None
    return frame_list

video_path = 'D:/ZLW_data/CAS(ME)^2/15_0101disgustingteeth.avi'
save_path = 'D:/ZLW_data/CAS(ME)^2/s15/15_0101disgustingteeth/'
convert_video2img(video_path, save_path, 1)
