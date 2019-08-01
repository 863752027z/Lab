import os


def get_path2(remain_dir):
    path1 = 'E:/CAS(ME)^2/rawvideo/'+ remain_dir + '/'
    path2 = 'E:/ZLW_generate_flow/' + remain_dir + '/'
    path_list1 = []
    path_list2 = []
    #counnt=1表示进入到第一个目录下
    count = 1
    for root, dirs, files in os.walk(path1):
        if count == 1:

            for i in range(len(files)):
                temp_path = path1 + files[i]
                path_list1.append(temp_path)
                temp_path = path2 + files[i][:-4]
                path_list2.append(temp_path)
            break
        count+=1
    return path_list1, path_list2

remain_dir = ['s21', 's22', 's23', 's24', 's25']
for i in range(len(remain_dir)):
    path_list1, path_list2 = get_path1(remain_dir[i])
    print(path_list1)
    print(path_list2)
    print('end')
