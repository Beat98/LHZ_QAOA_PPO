"""
Created on Wed Aug 22 12:33:46 2018

@author: kili

Creating Ising-formulations of random instances of different combinatorial problems.
"""

import numpy as np
import networkx as nx


def max_cut_random_ws_graph_to_ising(n_vertices, degree, p=0.5, different_vertices_prefactor=1,
                                     same_size_prefactor=None, seed=None):
    """
    creates graph with watts strogatz algorithm, for maxcut
    :param n_vertices: The number of nodes
    :param degree: Each node is connected to k nearest neighbors in ring topology
    :param p: The probability of adding a new edge for each edge
    :param different_vertices_prefactor:
    :param same_size_prefactor:
    :param seed: Seed for random number generator
    :return:
    """
    graph = nx.watts_strogatz_graph(n_vertices, degree, p, seed)
    return max_cut_graph_to_ising(graph, different_vertices_prefactor=different_vertices_prefactor,
                                  same_size_prefactor=same_size_prefactor)


def max_cut_random_nws_graph_to_ising(n_vertices, degree, p=0.5, different_vertices_prefactor=1,
                                      same_size_prefactor=None, seed=None):
    """
    creates graph with newman watts strogatz algorithm, for maxcut
    :param n_vertices: The number of nodes
    :param degree: Each node is connected to k nearest neighbors in ring topology
    :param p: The probability of adding a new edge for each edge
    :param different_vertices_prefactor:
    :param same_size_prefactor:
    :param seed: Seed for random number generator
    :return:
    """
    graph = nx.newman_watts_strogatz_graph(n_vertices, degree, p, seed)
    return max_cut_graph_to_ising(graph, different_vertices_prefactor=different_vertices_prefactor,
                                  same_size_prefactor=same_size_prefactor)


def max_cut_random_regular_graph_to_ising(n_vertices, degree, different_vertices_prefactor=1,
                                          same_size_prefactor=None, seed=None):
    """
    creates random regular graph for maxcut
    :param n_vertices: The number of nodes
    :param degree: Degree of the graph
    :param different_vertices_prefactor:
    :param same_size_prefactor:
    :param seed: Seed for random number generator
    :return:
    """
    graph = nx.random_regular_graph(degree, n_vertices, seed)
    return max_cut_graph_to_ising(graph, different_vertices_prefactor=different_vertices_prefactor,
                                  same_size_prefactor=same_size_prefactor)


def random_graph_partitioning_problem(n_vertices, degree, seed=None, different_vertices_prefactor=1,
                                      same_size_prefactor=None):
    return graph_partitioning_graph_to_ising(nx.newman_watts_strogatz_graph(n_vertices, degree, 0.5, seed), different_vertices_prefactor, same_size_prefactor)


def max_cut_graph_to_ising(graph, different_vertices_prefactor=-1):
    return graph_partitioning_graph_to_ising(graph, different_vertices_prefactor, 0)


def graph_partitioning_graph_to_ising(graph, different_vertices_prefactor=1, same_size_prefactor=None):
    """
    from http://arxiv.org/abs/1302.5843
    :param graph:
    :param different_vertices_prefactor: if positive cut the minimal number of edges,
    if negative cut the maximal number of edges
    :param same_size_prefactor:
    :return:
    """
    assert isinstance(graph, nx.Graph), 'graph must be of class NetworkX.Graph'

    k = max([d for n, d in graph.degree])  # maximum degree
    n_vertices = graph.order()

    same_size_prefactor_factor = min(2*k, n_vertices)/8

    if same_size_prefactor is None:
        same_size_prefactor = same_size_prefactor_factor*abs(different_vertices_prefactor)

    if same_size_prefactor < abs(different_vertices_prefactor)*same_size_prefactor_factor:
        Exception('same_size_prefactor too small')

    adj_m = np.array(nx.adj_matrix(graph, nodelist=range(n_vertices)).todense())

    return same_size_prefactor*np.ones((n_vertices, n_vertices)) + (1-adj_m)*different_vertices_prefactor/2


def MIS_graph_to_ising(graph):
    pass


def threeSAT_to_ising():
    pass


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # from investigations.n4maxcutgraphs import mc4graphs
    deg = 4
    numvertices = 9
    seedi = 2

    #graf = nx.random_regular_graph(deg, numvertices, seedi)
    graf = nx.newman_watts_strogatz_graph(numvertices, deg, 0.5, seed=seedi)
    # graf = nx.watts_strogatz_graph(numvertices, deg, 0.5, seed=seedi)

    # t = nx.Graph()
    # t.add_edge(1, 2)
    # t.add_edge(3, 4)
    # graf = t
    # graf = mc4graphs[3]

    if True:    # debug
        nx.draw(graf, with_labels=True)
        plt.axis('off')
        # plt.title('s: %d, deg: %d')
        plt.show()
        print()

    # bla = max_cut_graph_to_ising(graf, -1, 1)
    bla = max_cut_graph_to_ising(graf)
    bla = graph_partitioning_graph_to_ising(graf)

    from lhz.qutip_hdicts import Hdict_logical, Hdict_physical
    hd = Hdict_logical(bla)
    # hd = Hdict_physical(bla)
    from qutip import expect
    eenergies, evectors = hd['P'].eigenstates()
    from lhz.spinglass_utility import all_energies, translate_configuration_physical_to_logical

    eenergiess, evectorss = all_energies(bla[np.triu_indices_from(bla, 1)], True)
    eenergies2, evectors2 = [list(t) for t in zip(*sorted(zip(eenergiess, evectorss)))]

    ees = eenergies
    evs = evectors

    for i in range(len(np.where(ees == min(ees))[0])):
        if True:
            zrepres = expect(hd['z'], evs[i])
            zrepres = [-1 if zz < 0 else 1 for zz in zrepres]
            # zrepres = translate_configuration_physical_to_logical(zrepres)[0]
        else:
            zrepres = evs[i]

        print(zrepres)
        tgraf = graf.copy()

        remove_list = []
        for n1, n2 in tgraf.edges():
            if zrepres[n1] != zrepres[n2]:
                remove_list.append((n1, n2))

        for n1, n2 in remove_list:
            tgraf.remove_edge(n1, n2)

        nx.draw(tgraf, with_labels=True, node_color=['r' if zz > 0 else 'b' for zz in zrepres])
        plt.show()
