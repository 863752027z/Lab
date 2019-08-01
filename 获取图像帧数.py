cap = cv2.VideoCapture("2.avi")
# 获取FPS(每秒传输帧数(Frames Per Second))
fps = cap.get(cv2.CAP_PROP_FPS)
# 获取总帧数
totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
