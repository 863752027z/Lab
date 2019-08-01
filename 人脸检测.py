import cv2
video_path = 'E:/CAS(ME)^2/rawvideo/s19/19_0102eatingworms.avi'
detector = cv2.CascadeClassifier('E:/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(video_path)
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(total_frame)

count = 1
while True:
    #video_path = 'E:/hard.jpg'
    ret, img = cap.read()
    #img = img[0:420,0:500,:]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]
    img1 = img[y:y+h, x:x+w]

    cv2.imshow("img", img)
    k = cv2.waitKey(33)&0xff
    if k == 27:
        cv2.destroyAllWindows()
    print(count)
    count+=1

