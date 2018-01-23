import pandas as pd
import math
import numpy as np
from sklearn import metrics
from sklearn import mixture
from sklearn import cluster
from sklearn.cluster import KMeans
# import igraph
import scipy as sp
import scipy.linalg as linalg
import networkx as nx
import time
from collections import OrderedDict
# import community


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


def modularity(com, G):
    comm = [[] for i in range(max(com) + 1)]
    for m in range(len(com)):
        comm[com[m]].append(m)
    edges = G.edges()
    m = len(edges)
    du = G.degree()
    ret2 = 0.0
    for c in comm:
        bian = 0
        for x in c:
            for y in c:
                if x <= y:
                    if (x, y) in edges:
                        bian = bian + 1
                else:
                    if (y, x) in edges:
                        bian = bian + 1
        duHe = 0
        for x in c:
            duHe = duHe + du[x]
        tmp = bian * 1.0 / (2 * m) - (duHe * 1.0 / (2 * m)) * (duHe * 1.0 / (2 * m))
        ret2 = ret2 + tmp
    return ret2


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


def execute(W, e_normed, k):
    dict = OrderedDict()
    algorithms = ["k-means", "gmm", "sc", "dpgmm"]
    running_time = OrderedDict()
    for algo in algorithms:
        if algo is "k-means":
            start = time.clock()
            temp = KMeans(n_clusters=k, random_state=0).fit(W)
            clu = temp.labels_
            finish = time.clock()
            running_time["k-means"] = finish - start
        elif algo is "gmm":
            start = time.clock()
            temp = mixture.GaussianMixture(n_components=k + 1, covariance_type='full').fit(e_normed)
            clu = temp.predict(e_normed)
            finish = time.clock()
            running_time["gmm"] = finish - start
        elif algo is "sc":
            start = time.clock()
            temp = KMeans(n_clusters=k, random_state=0).fit(e_normed)
            clu = temp.labels_
            finish = time.clock()
            running_time["sc"] = finish - start
        elif algo is "dpgmm":
            start = time.clock()
            temp = mixture.BayesianGaussianMixture(n_components=k + 1, covariance_type='full').fit(e_normed)
            clu = temp.predict(e_normed)
            finish = time.clock()
            running_time["dpgmm"] = finish - start
        # print(clu)
        dict[algo] = clu

    return dict, running_time


def evaluate(truth, predicted, G):
    result = OrderedDict()
    for key in predicted:
        res = OrderedDict()
        NMI = metrics.normalized_mutual_info_score(truth, predicted[key])
        Modularity = Q(predicted[key], G)
        FMI = metrics.fowlkes_mallows_score(truth, predicted[key])
        ARI = metrics.adjusted_rand_score(truth, predicted[key])
        AMI = metrics.adjusted_mutual_info_score(truth, predicted[key])
        res["NMI"] = NMI
        res["Modularity"] = Modularity
        res["FMI"] = FMI
        res["ARI"] = ARI
        res["AMI"] = AMI
        result[key] = res
    return result


def truthLFR(dataset):
    truth_data = dataset + "/community.dat"
    data = pd.read_table(truth_data, header=None, sep='\t')
    l = data.shape[0]
    truth = [data[1][i] for i in range(l)]

    return truth


def output(all_res, running_times):
    print("****NMI***")
    for ds in all_res:
        print(ds, end='\t')
        for algorithm in all_res[ds]:
            print(all_res[ds][algorithm]["NMI"], end='\t')
        print()
    print("*******")
    print("****Modularity***")
    for ds in all_res:
        print(ds, end='\t')
        for algorithm in all_res[ds]:
            print(all_res[ds][algorithm]["Modularity"], end='\t')
        print()
    print("*******")
    print("****FMI***")
    for ds in all_res:
        print(ds, end='\t')
        for algorithm in all_res[ds]:
            print(all_res[ds][algorithm]["FMI"], end='\t')
        print()
    print("*******")
    print("****ARI***")
    for ds in all_res:
        print(ds, end='\t')
        for algorithm in all_res[ds]:
            print(all_res[ds][algorithm]["ARI"], end='\t')
        print()
    print("*******")
    print("****AMI***")
    for ds in all_res:
        print(ds, end='\t')
        for algorithm in all_res[ds]:
            print(all_res[ds][algorithm]["AMI"], end='\t')
        print()
    print("****Running times***")
    for ds in running_times:
        print(ds, end='\t')
        for algorithm in running_times[ds]:
            print(running_times[ds][algorithm]/20, end='\t')
        print()


def runRealWorld():
    datasets = ["karate", "football", "lesmis", "dolphins"]
    # datasets = ["karate"]
    all_res = OrderedDict()
    running_times = OrderedDict()
    for dataset in datasets:
        file_name = "data/realWorld/" + dataset + "/" + dataset + ".txt"
        W = pd.read_table(file_name, header=None, sep='\t').values
        G = networkG(W)
        truth = truthD(file_name)
        k = max(truth)
        Lbar = getNormLaplacian(W)
        # print(Lbar)
        e_normed = getKSmallestEigVec(Lbar, k)
        # print(e_normed)
        res, running_time = execute(W, e_normed, k)
        running_times[dataset] = running_time
        result = evaluate(truth, res, G)
        all_res[dataset] = result

    output(all_res, running_times)


def runLFR():
    # base = {"1285": 128, "5125": 512, "5123": 512, "5127": 512,
    #         "10246": 1024, "10244": 1024, "20485": 2048, "20483": 2048, "40966": 4096,
    #         "40963": 4096, "100004": 10000}
    base = OrderedDict()
    base["1285"] = 128
    base["5123"] = 512
    base["5125"] = 512
    base["5127"] = 512
    # base["10244"] = 1024
    # base["10246"] = 1024
    # base["10247"] = 1024
    # base["20483"] = 2048
    # base["20485"] = 2048
    # base["20487"] = 2048
    # base["40963"] = 4096
    # base["40965"] = 4096
    # base["40966"] = 4096
    # base["40967"] = 4096
    # base["100004"] = 10000
    # base["100006"] = 10000
    # base["100007"] = 10000
    all_res = OrderedDict()
    running_times = OrderedDict()
    for f in base:
        print(f + " running...")
        W = pd.read_table("data/LFR/" + f + "/network.dat", header=None, sep='\t').values
        comm = "data/LFR/" + f
        truth = truthLFR(comm)
        k = max(truth)
        G = nx.Graph()
        G.add_weighted_edges_from(W)
        adj = nx.to_numpy_matrix(G)
        Lbar = getNormLaplacian(adj)
        e_normed = getKSmallestEigVec(Lbar, k)
        # print(e_normed)
        res, running_time = execute(adj, e_normed, k)
        running_times[f] = running_time
        result = evaluate(truth, res, G)
        all_res[f] = result
        print(f + " finished")

    output(all_res, running_times)


if __name__ == "__main__":

    # runRealWorld()
    runLFR()