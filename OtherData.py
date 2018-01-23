import pandas as pd
import math
import numpy as np
# from sklearn import metrics
from sklearn import mixture
from sklearn import cluster
# import igraph
import scipy as sp
import scipy.linalg as linalg
import networkx as nx
import time
from sklearn.cluster import KMeans


def getNormLaplacian(W):
    """input matrix W=(w_ij)
    "compute D=diag(d1,...dn)
    "and L=D-W
    "and Lbar=D^(-1/2)LD^(-1/2)
    "return Lbar
    """
    d = [np.sum(row) for row in W]
    D = np.diag(d)
    L = D - W
    # Dn=D^(-1/2)
    Dn = np.power(np.linalg.matrix_power(D, -1), 0.5)
    Lbar = np.dot(np.dot(Dn, L), Dn)
    return Lbar


def getKSmallestEigVec(Lbar, k):
    """input
    "matrix Lbar and k
    "return
    "k smallest eigen values and their corresponding eigen vectors
    """
    eigval, eigvec = linalg.eig(Lbar)
    dim = len(eigval)
    # calculate the k smallest eigval
    dictEigval = dict(zip(eigval, range(0, dim)))
    kEig = np.sort(eigval)[0:k]
    ix = [dictEigval[k] for k in kEig]
    # return eigval[ix], eigvec[:, ix]
    return eigvec[:, ix]


def networkG(W):
    # g = igraph.Graph.Adjacency((W > 0).tolist())
    G = nx.from_numpy_matrix(W)
    return G


def truthD(dataset):
    truth_data = dataset.split('.')[0] + "_stan.txt"
    data = pd.read_table(truth_data, header=None, sep='\t')
    l = data.shape[0]
    truth = [data[0][i] for i in range(l)]
    return truth


def Q(comm, graph, weight='weight'):
    partition = {}
    if graph.has_node(0):
        for i in range(len(comm)):
            partition[i] = comm[i]
    else:
        for i in range(1, len(comm) + 1):
            partition[i] = comm[i - 1]

    if type(graph) != nx.Graph:
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def execute(e_normed, k):
    dict={}
    # gmm = mixture.mixtureGaussianMixture(n_components=n_samples, covariance_type='full').fit(e_normed)
    # print dpgmm.predict(e_normed)
    # n_gmm = max(gmm.predict(e_normed))
    # n_dpgmm = max(dpgmm.predict(e_normed))
    # print("real clusters: ", max(truth))
    # print("gmm clusters: ", n_gmm)
    # print("dpgmm clusters: ", n_dpgmm)
    begin = time.clock()
    dpgmm = mixture.BayesianGaussianMixture(n_components=k+1, covariance_type='full').fit(e_normed)
    end = time.clock()
    print("dpgmm running time(s)", end-begin)
    # res.append(dpgmm.predict(e_normed))
    dict["dpgmm"] = dpgmm.predict(e_normed)
    begin = time.clock()
    sc = KMeans(n_clusters=k, random_state=0).fit(e_normed)
    end = time.clock()
    print("sc running time(s)", end - begin)
    # print("sc running time(s)", end - begin)
    dict["sc"] = sc.labels_
    return dict


def evaluate(predicted, G):
    for key in predicted:
        print("-- "+key+" --")
        # print("NMI  ", metrics.normalized_mutual_info_score(truth, predicted[key]))
        print("Q    ", Q(predicted[key], G))
        # print("FMI  ", metrics.fowlkes_mallows_score(truth, predicted[key]))
        # print("ARI  ", metrics.adjusted_rand_score(truth, predicted[key]))
        # print("Accuracy  ", metrics.accuracy_score(truth, predicted[key]))
        # print("AMI  ", metrics.adjusted_mutual_info_score(truth, predicted[key]))
        # print("API  ", metrics.average_precision_score(truth, predicted[key]))
        print("-------")


if __name__ == "__main__":

    # datasets = ["email", "jazz", "polbooks", "protein"]
    # datasets ={"email":23, "jazz":5, "polbooks":3, "protein":13}
    datasets ={"AstrophysicsCollaborations":1200, "internet":50, "polblogs":280,
            "polbooks":4, "power":50, "jazz":5,
            "email":23, "PGPgiantcompo":98, "celegans_metabolic":10}
    for key in datasets:
        file_name = "data/realWorld/other/"+key+".net"
        print("------------------"+key+"------------------")
        # W = pd.read_table(file_name, header=None, sep='\t').values
        # G = networkG(W)
        G = nx.read_pajek(file_name)
        W = nx.to_numpy_matrix(G)
        k = datasets[key]
        Lbar = getNormLaplacian(W)
        e_normed = getKSmallestEigVec(Lbar, k)
        res = execute(W, e_normed, k)
        # print(res)
        evaluate(res, G)
        print("----------------" + key + " ends -----------------")
        print()



