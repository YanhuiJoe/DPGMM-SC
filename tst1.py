from __future__ import division

from itertools import product

import networkx as nx

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


def modularity(G, communities, weight='weight'):
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


# G = nx.barbell_graph(3, 0)
# # print(G.edges)
# q = modularity(G, [[0, 1], [2, 3, 4, 5]])
# print(q)

comm = [0, 1, 0, 2, 0, 2, 1]
communities = [[] for i in range(max(comm)+1)]
for i in range(len(comm)):
    communities[comm[i]].append(i)
print(communities)