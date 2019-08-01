import cv2
import numpy as np
import math

def calcOpticalFlow(pre_frame, current_frame, p0):
    old_gray = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    lk_params = dict(winSize=(15,15),
                 maxLevel=2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_old = p0[st==1]
    good_new = p1[st==1]
    return p0,p1,st

def draw_point(P, img):
    for i in range(len(P)):
        temp_point = (P[i][0],P[i][1])
        cv2.circle(img, temp_point, 2, color=(0, 0, 255))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i + 1), temp_point, font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

frame = []
frame_point = []
base_path = 'E:/optical_flow/'
for i in range(14):
    path = base_path + str(i+136) + '.jpg'
    img = cv2.imread(path)
    img = img[240:,250:]
    frame.append(img)
img = frame[0]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7)
p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
#point = np.array([p0[2]], dtype = np.float32)
#p0 = point

pos = np.array([])
mod = np.array([])
angle = np.array([])
count = 0

print(len(frame)-1)
for i in range(len(frame)-1):
    count += 1
    pre_frame = frame[i]
    current_frame = frame[i+1]
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
        v = [good_new[j][0]-good_old[j][0], good_new[j][1]-good_old[j][1]]
        temp_m = np.sqrt(np.dot(np.transpose(v), v))
        temp_a = math.atan2(good_old[j][1], good_old[j][0])
        m.append(temp_m)
        p.append(good_old[j])
        a.append(temp_a)

    m = np.array(m)
    #p = np.array([p])
    a = np.array(a)
    
    if count == 1:
        mod = np.empty([good_old.shape[0], 1], dtype=np.float32)
        mod = np.column_stack((mod, m))
        mod = np.delete(mod, 0, axis=1)
        '''
        pos = np.empty([good_old.shape[0],1, 2], dtype=np.float32)
        pos = np.column_stack((pos, p))
        pos = np.delete(pos, 0, axis=1)
        '''
        angle = np.empty([good_old.shape[0], 1], dtype=np.float32)
        angle = np.column_stack((angle, a))
        angle = np.delete(angle, 0, axis=1)
    else:
        if mod.any():
            mod = np.column_stack((mod, m))
        '''
        if pos.any():
            pos = np.column_stack((pos, p))
        '''
        if angle.any():
            angle = np.column_stack((angle, a))

    print(i + 1)
    print(good_old)
    print(st)
    print(mod)
    print(angle)
    #print(pos)
    draw_point(good_new, frame[i + 1])
    cv2.imshow('img', frame[i + 1])
    cv2.waitKey(0)




'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
print(gray.shape)
print(gray1.shape)

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7)
p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

p0, p1, st = calcOpticalFlow(gray, gray1, p0)
print(p0)
print(p1)
'''