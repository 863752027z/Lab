import re
import os

s1 = '006-10000'
s2 = '006-01000'
s3 = '006-00100'
s4 = '006-00010'
s5 = '006-00001'
s6 = '006-00000'
s1 = s1[4:]
mode = '006[1-9][0-9]{0,4}'
num = re.findall(mode, s2)

mode = '[1-9][0-9]{0,4}'
path = 'D:/SAMM/006/1_part1/'
path = 'D:/SAMM/006/1_part2/'
first = None
count = 0
for l in os.listdir(path):
    count += 1
    old_name = l
    if count == 1:
        first = l
        new_name = '0.jpg'
    else:
        l = l[4:]
        num = re.findall(mode, l)
        new_name = ''.join(x for x in num) + '.jpg'
    print(old_name)
    print(new_name)
    old_path = path + old_name
    new_path = path + new_name
    os.rename(old_path, new_path)

