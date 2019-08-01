import os
def get_file_name(file_path):
    for root, dirs, files in os.walk(file_path):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件


def get_path():
    path1 = 'E:/CAS(ME)^2/rawvideo/s15/'
    path2 = 'E:/ZLW_generate_flow/s15/'
    path_list1 = []
    path_list2 = []
    #counnt=1表示进入到第一个目录下
    count = 1
    for root, dirs, files in os.walk(path1):
        if count == 1:
            print(root)
            print(dirs)
            print(files)
            for i in range(len(files)):
                temp_path = path1 + files[i]
                path_list1.append(temp_path)
                temp_path = path2 + files[i][:-4]
                path_list2.append(temp_path)
            break
        count+=1
    return path_list1, path_list2

path_list1, path_list2 = get_path()
print('***************************')
print(path_list1)
print('***************************')
print(path_list2)





