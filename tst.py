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
from collections import OrderedDict
from sklearn import mixture
from itertools import product


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

def is_partition(G, partition):
    """Returns True if and only if `partition` is a partition of
    the nodes of `G`.
    A partition of a universe set is a family of pairwise disjoint sets
    whose union equals the universe set.
    `G` is a NetworkX graph.
    `partition` is a sequence (not an iterator) of sets of nodes of
    `G`.
    """
    return all(sum(1 if v in c else 0 for c in partition) == 1 for v in G)


def modularity(G, comm, weight='weight'):
    r"""Returns the modularity of the given partition of the graph.
    Modularity is defined in [1]_ as
    .. math::
        Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_ik_j}{2m}\right)
            \delta(c_i,c_j)
    where *m* is the number of edges, *A* is the adjacency matrix of
    `G`, :math:`k_i` is the degree of *i* and :math:`\delta(c_i, c_j)`
    is 1 if *i* and *j* are in the same community and 0 otherwise.
    Parameters
    ----------
    G : NetworkX Graph
    communities : list
        List of sets of nodes of `G` representing a partition of the
        nodes.
    Returns
    -------
    Q : float
        The modularity of the paritition.
    Raises
    ------
    NotAPartition
        If `communities` is not a partition of the nodes of `G`.
    Examples
    --------
    G = nx.barbell_graph(3, 0)
    nx.modularity(G, [{0, 1, 2}, {3, 4, 5}])
    0.35714285714285704
    References
    ----------
    .. [1] M. E. J. Newman *Networks: An Introduction*, page 224.
       Oxford University Press, 2011.
    """
    communities = [[] for i in range(max(comm) + 1)]
    if G.has_node(0):
        for i in range(len(comm)):
            communities[comm[i]].append(i)
    else:
        # print(G.edges)
        for i in range(len(comm)):
            communities[comm[i]].append(i+1)

    if not is_partition(G, communities):
        raise TypeError("NotAPartition")

    multigraph = G.is_multigraph()
    directed = G.is_directed()
    m = G.size(weight=weight)
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        norm = 1 / m
    else:
        out_degree = dict(G.degree(weight=weight))
        in_degree = out_degree
        norm = 1 / (2 * m)

    def val(u, v):
        try:
            if multigraph:
                w = sum(d.get(weight, 1) for k, d in G[u][v].items())
            else:
                w = G[u][v].get(weight, 1)
        except KeyError:
            w = 0
        # Double count self-loops if the graph is undirected.
        if u == v and not directed:
            w *= 2
        return w - in_degree[u] * out_degree[v] * norm

    Q = sum(val(u, v) for c in communities for u, v in product(c, repeat=2))
    return Q * norm


def execute(W, e_normed, k):
    dict = OrderedDict()
    algorithms = ["k-means", "gmm", "sc", "dpgmm"]
    # gmm = mixture.mixtureGaussianMixture(n_components=n_samples, covariance_type='full').fit(e_normed)
    # print dpgmm.predict(e_normed)
    # n_gmm = max(gmm.predict(e_normed))
    # n_dpgmm = max(dpgmm.predict(e_normed))
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
            temp = mixture.GaussianMixture(n_components=k+1, covariance_type='full').fit(e_normed)
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
            temp = mixture.BayesianGaussianMixture(n_components=k+1, covariance_type='full').fit(e_normed)
            clu = temp.predict(e_normed)
            finish = time.clock()
            running_time["sc"] = finish - start
        # print(clu)
        dict[algo] = clu
    # begin = time.clock()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=k+1, covariance_type='full').fit(e_normed)
    # end = time.clock()
    # print("dpgmm running time(s)", end-begin)
    # # res.append(dpgmm.predict(e_normed))
    # dict["dpgmm"] = dpgmm.predict(e_normed)
    # begin = time.clock()
    # sc = KMeans(n_clusters=k, random_state=0).fit(e_normed)
    # end = time.clock()
    # print("sc running time(s)", end - begin)
    # # print("sc running time(s)", end - begin)
    # dict["sc"] = sc.labels_
    return dict, running_time


def evaluate(predicted, G):
    result = OrderedDict()
    for key in predicted:
        res = OrderedDict()
        Modularity = modularity(G, predicted[key])
        m = nx.directed_modularity_matrix
        res["Modularity"] = Modularity
        result[key] = res
    return result


def output(all_res, running_times):
    print("****Modularity***")
    for ds in all_res:
        print(ds, end='\t')
        for algorithm in all_res[ds]:
            print(all_res[ds][algorithm]["Modularity"], end='\t')
        print()
    print("*******")
    print("****Running times***")
    for ds in running_times:
        print(ds, end='\t')
        for algorithm in running_times[ds]:
            print(running_times[ds][algorithm], end='\t')
        print()


if __name__ == "__main__":

    # datasets = ["email", "jazz", "polbooks", "protein"]
    datasets ={"polblogs":4}
    # datasets ={"AstrophysicsCollaborations":1200, "internet":50, "polblogs":280,
    #         "polbooks":4, "power":50, "jazz":5,
    #         "email":23, "PGPgiantcompo":98, "celegans_metabolic":10}
    all_res = OrderedDict()
    running_times = OrderedDict()
    for key in datasets:

        file_name = "data/realWorld/other/"+key+".net"
        print("----" + key + " running...----")
        # W = pd.read_table(file_name, header=None, sep='\t').values
        # G = networkG(W)
        G = nx.read_pajek(file_name)
        W = nx.to_numpy_matrix(G)
        k = datasets[key]
        Lbar = getNormLaplacian(W)
        e_normed = getKSmallestEigVec(Lbar, k)
        res, running_time = execute(W, e_normed, k)
        # print(res)
        result = evaluate(res, G)
        all_res[key] = result
        print("----" + key + " finishied...----")
    output(all_res, running_times)
