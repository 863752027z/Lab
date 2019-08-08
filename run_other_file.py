path = 'D:/file.txt'

print('hello world')
with open(path, 'a+') as f:
    f.write('hello world')
    f.write('\n')
