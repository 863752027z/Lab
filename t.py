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
print(st.reshape(-1))
