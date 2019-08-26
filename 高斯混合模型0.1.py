import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

#产生实验数据
sns.set()

X, y_true = make_blobs(n_samples=4000, centers=2,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1]
#交换列是为了方便画图

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2).fit(X)
labels = gmm.predict(X)
print(labels)
print(labels.shape)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()

"""
probs = gmm.predict_proba(X)
#print(probs[:5].round(3))
size = 50 * probs.max(1) ** 2  #平方放大概率的差异
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)
"""