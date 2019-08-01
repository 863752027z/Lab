def rw_img(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(total_frame_number)
    print(fps)
    COUNT = 0
    while COUNT < total_frame_number:
        # 一帧一帧图像读取
        ret, frame = cap.read()
        # 把每一帧图像保存成jpg格式（这一行可以根据需要选择保留）
        #cv2.imwrite('E:/save_img/' + str(COUNT) + '.jpg', frame)
        # 显示这一帧地图像
        cv2.imshow('video', frame)
        COUNT = COUNT + 1
        # 延时一段33ms（1s➗30帧）再读取下一帧，如果没有这一句便无法正常显示视频
        cv2.waitKey(33)
        k = cv2.waitKey(0)&0xff
        if k == 27:
            break
    cap.release()