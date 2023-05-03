import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import k_hop_subgraph,to_networkx
import ot
import time


#rng = np.random.RandomState(42)

def get_dataset(dataset_name):
    """ 
    Function that returns the dataset and the number of classes for a given
    dataset name
    
    Input:
        dataset_name: name of the dataset
    Output: 
        dataset: dataset
        n_classes: number of classes

    """

    if dataset_name=='Citeseer':
        dataset=Citeseer_data()
        n_classes=6

    elif dataset_name=='Toy_graph':
        dataset=torch.load('data/toy_graph1.pt')
        n_classes=3

    return dataset,n_classes

def get_filenames(dataset_name,method,seed=None):

    if seed is None:

        filename_save=os.path.join( 'results',method,"{}.pkl".format(dataset_name))
        filename_best_model=os.path.join( 'results',method,"{}_best_valid.pkl".format(dataset_name))   

    else:   
        filename_save=os.path.join( 'results',method,"{}_seed{}.pkl".format(dataset_name,seed))
        filename_best_model=os.path.join( 'results',method,"{}_seed{}_best_valid.pkl".format(dataset_name,seed))
    return filename_save, filename_best_model


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
    C=C.to_dense()
    return C+C.T


def subgraph(x,edge_index,node_idx, order):
    """
    Computes the edges and nodes of a subgraph center at node_idx of order k
    C : adjacency matrix of the graph
    x : features of the nodes of the graph
    node_idx : index of the node to center the subgraph
    order : order of te subgraph (number of neigbours)
    """
    sub_G=k_hop_subgraph(node_idx,order,edge_index=edge_index,relabel_nodes=True) 
    x_sub=x[sub_G[0]]
    edges_sub=sub_G[1]
    central_node_index=sub_G[2]
    return x_sub,edges_sub,central_node_index



def distance_to_template(x,edge_index,x_T,C_T,alpha,q,k=1):
    """
    Computes the OT distance between each subgraphs of order k of G and the templates
    x : node features of the graph
    edge_index : edge indexes of the graph
    x_T : list of the node features of the templates
    C_T : list of the adjacency matrices of the templates 
    alpha : trade-off parameter for fused gromov-wasserstein distance
    k : number of neighbours in the subgraphs
    """
    n=len(x)       #number of nodes in the graph
    n_T=len(x_T)   #number of templates
    n_feat=len(x[0])
    n_feat_T=len(x_T[0][0])
    if not n_feat==n_feat_T:
        raise ValueError('the templates and the graphs must have the same number of features')
    distances=torch.zeros(n,n_T)
    for i in range(n):
        x_sub,edges_sub,central_node_index=subgraph(x,edge_index,i,k)
        x_sub=x_sub.reshape(len(x_sub),n_feat)  #reshape pour utiliser ot.dist      
        n_sub=len(x_sub)
        if n_sub>1:    #more weight on central node
          p=torch.ones(n_sub)*1/((n_sub-1)*(k+2))
          p[central_node_index]=(k+1)/(k+2)  
        else:          #if the node is isolated
          p=torch.ones(1)
        C_sub=graph_to_adjacency(n_sub,edges_sub).type(torch.float)    
        for j in range(n_T):
          template_features=x_T[j].reshape(len(x_T[j]),n_feat_T)   #reshape pour utiliser ot.dist
          M=ot.dist(x_sub,template_features).clone().detach().requires_grad_(True)
          M=M.type(torch.float)  #cost matrix between the features of the subgraph and the template
          dist=ot.gromov.fused_gromov_wasserstein2(M, C_sub, C_T[j], p, q[j],alpha=alpha,symmetric=True,max_iter=100)    
          distances[i,j]=dist
    return distances
