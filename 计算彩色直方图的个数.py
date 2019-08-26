# 彩色直方图 和 灰度直方图 原理是一样的,彩色直方图需要计算三个通道
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'D:/ZLW_data/SAMM/flow_cut_eye/0.jpg'
img = cv2.imread(path, 1)
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

