import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import k_hop_subgraph,to_networkx
import ot

def visualize_graph(G, color='b'):
    """"
    visualisation of a torch.geometric graph
    """
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    G = to_networkx(G, to_undirected=True)
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, cmap="Set2")
    plt.show()

def plot_graph(x, C, binary=True, color='C0', s=None):
    for j in range(C.shape[0]):
        for i in range(j):
            if binary:
                if C[i, j] > 0:
                    plt.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], alpha=0.2, color='k')
            else:  # connection intensity proportional to C[i,j]
                plt.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], alpha=C[i, j], color='k')

    plt.scatter(x[:, 0], x[:, 1], c=color, s=s, zorder=10, edgecolors='k', cmap='tab10', vmax=9)

def get_sbm(n, nc, ratio, P):
    torch.manual_seed(32)
    nbpc = torch.round(n * ratio).type(torch.int64)
    n =  torch.sum(nbpc).item()
    C =  torch.zeros(n, n)
    for c1 in range(nc):
        for c2 in range(c1 + 1):
            if c1 == c2:
                for i in range( torch.sum(nbpc[:c1]),  torch.sum(nbpc[:c1 + 1])):
                    for j in range( torch.sum(nbpc[:c2]), i):
                        if torch.rand(1) <= P[c1, c2]:
                            C[i, j] = 1
            else:
                for i in range( torch.sum(nbpc[:c1]),  torch.sum(nbpc[:c1 + 1])):
                    for j in range( torch.sum(nbpc[:c2]),  torch.sum(nbpc[:c2 + 1])):
                        if torch.rand(1) <= P[c1, c2]:
                            C[i, j] = 1
    return C + C.T


def adjacency_to_graph(C,F):
    """
    Returns a torch_geometric graph given binary adjacency matrix and features
    C : adjacency matrix of the graph
    F : features of the nodes of the graph
    """
    edges=torch.where(C==1)
    edge_index=torch.stack(edges)
    return GraphData(x=F,edge_index=edge_index)


def graph_to_adjacency(n,edges): 
    """"
    adjacency matrix of a graph given its nodes and edges in a torch.geometric format
    n : number of nodes
    edges : edges in the format [[senders],[receivers]]

    Returns: sparse adjacency matrix C
    """
    C=torch.sparse_coo_tensor(edges, np.ones(len(edges[0])),size=(n, n))
    return C.to_dense()


def subgraph(C,F,node_idx, order):
    """
    Computes the edges and nodes of a subgraph center at node_idx of order k
    C : adjacency matrix of the graph
    x : features of the nodes of the graph
    node_idx : index of the node to center the subgraph
    order : order of te subgraph
    """
    G=adjacency_to_graph(C,F)
    sub_G=k_hop_subgraph(node_idx,order,edge_index=G.edge_index,relabel_nodes=True) 
    F_sub=F[sub_G[0]]
    edges_sub=sub_G[1]
    C_sub=graph_to_adjacency(len(sub_G[0]),edges_sub).type(torch.float64)
    return C_sub,F_sub


def distance_to_template(C,F_C,T,F_T,k,n_feat,alpha):
    """
    Computes the OT distance between each subgraphs of order k of G and the templates
    C : adjacency matrix of the graph
    F_C : features of the nodes of the graph
    T : list of adjacency matrices of the templates
    F_T : list of the features of the templates nodes
    k : number of neighbours in the subgraphs
    n_feat : dimension of the features
    alpha : trade-off parameter for fused gromov-wasserstein distance
    """
    n=C.shape[0]
    n_T=len(T)
    distances=torch.zeros(n,n_T)

    for i in range(n):
        print(i)
        C_sub,F_sub=subgraph(C,F_C,i,k)
        for j in range(n_T):
          F_sub=F_sub.reshape(len(F_sub),n_feat)  #reshape pour utiliser ot.dist
          template_features=F_T[j].reshape(len(F_T[j]),n_feat)   #reshape pour utiliser ot.dist
          M=torch.tensor(ot.dist(F_sub,template_features))  #cost matrix between the features of the subgraph and the template
          n_sub=len(F_sub)
          n_template=len(F_T[j]) 
          p=torch.ones(n_sub)/n_sub
          q=torch.ones(n_template)/n_template
          dist=ot.gromov.fused_gromov_wasserstein2(M, C_sub, T[j], p, q,alpha=alpha)
          distances[i,j]=dist
    return distances

