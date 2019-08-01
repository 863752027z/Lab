import itertools

list = []
for i in range(7):
    list.append(i+1)
print(list)
for i in itertools.combinations(list, 3):
    print(i)
