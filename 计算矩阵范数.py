import numpy as np

def calcNorm2(m):
    mT = np.transpose(m)
    mTm = np.dot(mT, m)
    value, vector = np.linalg.eig(mTm)
    max_value = np.max(value)
    norm2 = np.sqrt(max_value)
    return norm2

print("###########矩阵范数#########")
a = np.arange(12).reshape(3,4)
print("矩阵a为：")
print(a)
print("F范数",np.linalg.norm(a,ord = 'fro'),"矩阵元素绝对值的平方和再开平方")
print("1范数",np.linalg.norm(a,ord = 1),"列和范数，即所有矩阵列向量绝对值之和的最大值")
print("2范数",np.linalg.norm(a,ord = 2),"谱范数，即ATA矩阵的最大特征值的开平方")
print("无穷范数",np.linalg.norm(a,ord = np.inf),"行和范数，即所有矩阵行向量绝对值之和的最大值")


print(calcNorm2(a))