import pandas as pd
import math
import numpy as np
from sklearn import metrics
from sklearn import mixture
from sklearn import cluster
import scipy as sp
import scipy.linalg as linalg
import networkx as nx
import matplotlib.pyplot as plt

datasets = ["dolphins","football","karate","lesmis"]


data = pd.read_table("truth.txt",header=None,sep='\t')
len = data.shape[0]
truth = [data[0][i] for i in range(len)]
W = pd.read_table("football.txt",header=None,sep='\t').values

n_samples = W.shape[0]

## n : a crucial param
print("number of Samples: ", n_samples)

n = int(math.sqrt(n_samples/2))
# n = int(n_samples/2)
print("selected number of Samples: ", n)

print(W.shape)
assert W.shape[0] == W.shape[1]
nn = W.shape[0]
d = W.sum(axis=1)
D = np.diag(d)
L = D - W
L = L / L.max()
X, s, V = np.linalg.svd(L, full_matrices=False)
evecs = X[:, (nn - n):nn]
e_normed = evecs / np.tile(evecs.max(axis=1), (n, 1)).transpose()
# k = autosp.predict_k(W)
# print(k)
# print(e_normed)
gmm = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(e_normed)
dpgmm = mixture.BayesianGaussianMixture(n_components=n,covariance_type='diag').fit(e_normed)

n_gmm = max(gmm.predict(e_normed))
n_dpgmm = max(dpgmm.predict(e_normed))

print("gmm clusters: ", n_gmm)
print("dpgmm clusters: ",n_dpgmm)

sc = cluster.spectral_clustering(W, n_clusters=n_dpgmm)
print(sc)
# labels = sc.labels_
# kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(e_normed)
# labels = kmeans.labels_

# print(truth)
print("================NMI=================")
# print("k-means: ", metrics.normalized_mutual_info_score(truth, labels))
print("sc: ", metrics.normalized_mutual_info_score(truth, sc))
print("gmm: ", metrics.normalized_mutual_info_score(truth, gmm.predict(e_normed)))
print("dpgmm: ", metrics.normalized_mutual_info_score(truth, dpgmm.predict(e_normed)))
print("====================================")