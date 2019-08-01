from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


X, y_true = make_blobs(n_samples=400,n_features=4, centers=2,
                       cluster_std=0.60, random_state=0)
#print(X)
print(X.shape)
print(X)
#plt.scatter(X[:,0],X[:,1], c=y_true,cmap= plt.cm.spring, edgecolor = 'k')

gmm = GaussianMixture(n_components=2).fit(X)
#print(gmm)
#print(gmm)
labels = gmm.predict(X)
print(labels)
probs = gmm.predict_proba(X)
print(probs)
print(probs.shape)
p1 = 0
p2 = 0

#print(labels)
#plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
#plt.show()
