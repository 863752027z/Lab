import cv2
from PIL import Image, ImageSequence
import os

frame_list = []
path = 'D:/AU12.gif'
im = Image.open(path)
save_path = 'D:/save_img/'
for i, frame in enumerate(ImageSequence.Iterator(im), 1):
    save_path = save_path + str(i) + '.jpeg'
    frame = frame.convert('RGB')
    frame.save(save_path, "jpeg")
    frame_list.append(save_path)
    save_path = 'D:/save_img/'
print(frame_list)

base_path = 'D:/simile'
if not os.path.exists(base_path):
    os.mkdir(base_path)
for i in range(12):
    save_path = base_path + '/' + str(i+1) + '.jpg'
    img = cv2.imread(frame_list[i])
    img = img[50:100, 80:220, :]
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imwrite(save_path, img)
    save_path = None

'''
for i in range(len(frame_list)):
    print(frame_list[i])
    img = cv2.imread(frame_list[i])
    cv2.imshow('img', img)
    cv2.waitKey(0)
'''
