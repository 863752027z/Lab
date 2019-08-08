import numpy as np
import cv2

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

    return True if sinsc%2==1 else  False
def draw_line(img, ptStart, ptEnd):
    ptStart = (ptStart[0], ptStart[1])
    ptEnd = (ptEnd[0], ptEnd[1])
    thickness = 1
    cv2.line(img, ptStart, ptEnd, (0, 255, 0), thickness)#绿色，3个像素宽度

def draw_point(P, img):
    for i in range(len(P)):
        temp_point = (P[i][0], P[i][1])
        cv2.circle(img, temp_point, 1, color=(0, 0, 255))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i), temp_point, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

picture = np.zeros((500, 500, 3))
P = [[20, 40], [40, 60], [60, 88], [80, 100], [100, 300]]
Point = [[20, 100], [30, 60], [70, 130], [30, 30], [80, 200], [300, 300]]
poly =[]
for i in range(len(P)-1):
    temp = []
    ptStart = P[i]
    ptEnd = P[i+1]
    draw_line(picture, ptStart, ptEnd)
    poly.append([ptStart, ptEnd])
draw_line(picture, P[-1], P[0])
draw_point(Point, picture)

st = []
not_in = []
is_in = []
i = 0
for p in Point:
    if(isPoiWithinPoly(p, poly)):
        is_in.append(i)
        st.append(1)
    else:
        not_in.append(i)
        st.append(0)
    i += 1
print('不在里面的点:')
print(not_in)
print('在里面的点:')
print(is_in)
print(st)


cv2.imshow('image', picture)
cv2.waitKey(0)
