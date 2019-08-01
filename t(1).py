import numpy as np
a = np.array([[1,2],[4,5]])
i = np.array([5,6])
x = np.empty([2,1], dtype=np.float32)
print(a)
print(i)

c = np.column_stack((x,i))
print(c)
a = [1,2]
d = np.sqrt(np.dot(np.transpose(a), a))
print(d)

print('start')

st = np.array([[0], [1]])
print(st)
print(st.shape)
x = np.array([[[1,2]],[[4,5]]])
print(x)
print(x[st==0])
print('st')
print(st.flatten())
print(st.reshape(-1))
st = st.reshape(-1)
print(st)

pos = np.array([[1, 1], [1, 2]])
p = [[1,3]]
pos = np.concatenate([pos,p])
print(pos)

#print(pos[st == 1])

w = np.empty([0, 1])
print(w)

temp = np.array([[1]])
w = np.concatenate([w, temp])
w = np.concatenate([w, temp])
print(w[st==1])
print(st)
print(w)

print('data_set')
data_set = [[[0, 2, 3], [1, 2, 3]],
            [[0, 3, 4], [1, 3, 4]]]
data_set = np.delete(data_set, 0, axis = 1)
print(data_set)

