import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import k_hop_subgraph,to_networkx
import os
import ot
import time
from data.convert_datasets import Citeseer_data, Cornell
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path as function_shortest_path

def get_dataset(dataset_name):
    """ 
    Function that returns the dataset and the number of classes for a given
    dataset name
    
    Input:
        dataset_name: name of the dataset
    Output: 
        dataset: dataset
        n_classes: number of classes
        n_features: number of node features
        test_graph: wether there is a separate graph for testing

    """

    if dataset_name=='Citeseer':
        dataset=Citeseer_data()
        n_classes=6
        n_features=dataset.num_features
        test_graph=False
        graph_type='single_graph'

    elif dataset_name=='Toy_graph_single':
        dataset_train=torch.load('data/toy_single_train.pt')
        dataset_test=torch.load('data/toy_single_test.pt')
        dataset=[dataset_train,dataset_test]
        n_classes=3
        n_features=dataset_train.num_features
        test_graph=True
        graph_type='single_graph'

    elif dataset_name=='Toy_graph_multi':
        dataset=torch.load('data/toy_multi_graph.pt')
        n_classes=3
        n_features=dataset[0].num_features
        test_graph=None
        graph_type='multi_graph'

    elif dataset_name=='mutag':
        dataset=torch.load('data/mutag.pt')
        n_classes=7
        n_features=dataset[0].num_features
        test_graph=None
        graph_type='multi_graph'

    elif dataset_name=='cornell':
       dataset=torch.load('data/cornell.pt')
       n_classes=5
       n_features=dataset.num_features
       test_graph=False
       graph_type='single_graph'

    return dataset,n_classes,n_features, test_graph, graph_type

def get_filenames(dataset_name,method,lr,n_temp,n_nodes,alpha0,local_alpha,k,dropout,shortest_path,seed=None):

    if seed is None:
        filename_save=os.path.join( 'results',method,"{}.pkl".format(dataset_name))
        filename_best_model=os.path.join( 'results',method,"{}_best_valid.pkl".format(dataset_name)) 
        filename_visus=os.path.join( 'results',method,"{}_visus.pkl".format(dataset_name)) 

    else:   
        filename_save=os.path.join( 'results',method,"{}_seed{}_lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_localalpha{}_dropout{}_shortest_path{}.pkl".format(dataset_name,seed,lr,n_temp,n_nodes,alpha0,k,local_alpha,dropout,shortest_path))
        filename_best_model=os.path.join( 'results',method,"{}_seed{}_lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_localalpha{}_dropout{}_shortest_path{}_best_valid.pkl".format(dataset_name,seed,lr,n_temp,n_nodes,alpha0,k,local_alpha,dropout,shortest_path))
        filename_visus=os.path.join( 'results',method,"{}_seed{}_lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_localalpha{}_dropout{}_shortest_path{}_visus.pkl".format(dataset_name,seed,lr,n_temp,n_nodes,alpha0,k,local_alpha,dropout,shortest_path))

    return filename_save, filename_best_model, filename_visus


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


def graph_to_adjacency(n,edges,shortest_path): 
    """"
    adjacency matrix of a graph given its nodes and edges in a torch.geometric format
    n : number of nodes
    edges : edges in the format [[senders],[receivers]]

    Returns: sparse adjacency matrix C
    """
    C=torch.sparse_coo_tensor(edges, np.ones(len(edges[0])),size=(n, n))
    C=C.to_dense()
    C=C+C.T
    if not shortest_path:
        return C
    else:
        graph=csr_matrix(C)
        dist_matrix=function_shortest_path(graph)
        return torch.Tensor(dist_matrix)

   


def subgraph(x,edge_index,node_idx, order,num_nodes):
    """
    Computes the edges and nodes of a subgraph center at node_idx of order k
    C : adjacency matrix of the graph
    x : features of the nodes of the graph
    node_idx : index of the node to center the subgraph
    order : order of te subgraph (number of neigbours)
    """

    sub_G=k_hop_subgraph(node_idx,order,edge_index=edge_index,relabel_nodes=True,num_nodes=num_nodes) 
    x_sub=x[sub_G[0]]
    edges_sub=sub_G[1]
    central_node_index=sub_G[2]
    return x_sub,edges_sub,central_node_index


def distance_to_template(x,edge_index,x_T,C_T,alpha,q,k,local_alpha,shortest_path):
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

    #normalize q for gromov-wasserstein
    q=F.normalize(q,p=1,dim=1)
    
    if not n_feat==n_feat_T:
        raise ValueError('the templates and the graphs must have the same number of features')
    
    distances=torch.zeros(n,n_T)
    for i in range(n):
        x_sub,edges_sub,central_node_index=subgraph(x,edge_index,i,k,n)
        x_sub=x_sub.reshape(len(x_sub),n_feat)  #reshape pour utiliser ot.dist      
        n_sub=len(x_sub)

        if n_sub>1:    #more weight on central node
          val=(1-(k+1)/(k+2))/(n_sub-1)
          p=torch.ones(n_sub)*val
          p[central_node_index]=(k+1)/(k+2)
          p=F.normalize(p,p=1,dim=0)    #normalize for gromov-wasserstein

        else:          #if the node is isolated
          p=torch.ones(1)
          p=F.normalize(p,p=1,dim=0)  #normalize p for gromov-wasserstein

        C_sub=graph_to_adjacency(n_sub,edges_sub,shortest_path).type(torch.float)
 
        for j in range(n_T):
          
          template_features=x_T[j].reshape(len(x_T[j]),n_feat_T)   #reshape pour utiliser ot.dist
          M=ot.dist(x_sub,template_features).clone().detach().requires_grad_(True)
          M=M.type(torch.float)  #cost matrix between the features of the subgraph and the template

          #more normalization
          qj=q[j]/torch.sum(q[j])
          p=p/torch.sum(p)

          #ensure that p and q have the same sum
          p_nump=p.numpy()
          p_nump=np.asarray(p_nump, dtype=np.float64)
          sum_p=p_nump.sum(0)
          q_nump=qj.detach().numpy()
          q_nump=np.asarray(q_nump, dtype=np.float64)
          sum_q=q_nump.sum(0)
          if not abs(sum_q-sum_p) < np.float64(1.5 * 10**(-7)):
              if sum_q>sum_p:
                  p[0]+=abs(sum_q-sum_p)
              else:
                  qj[0]+=abs(sum_q-sum_p)   
          if local_alpha: 
             dist=ot.gromov.fused_gromov_wasserstein2(M, C_sub, C_T[j], p, qj,alpha=alpha[i],symmetric=True,max_iter=100) 
          else:
             dist=ot.gromov.fused_gromov_wasserstein2(M, C_sub, C_T[j], p, qj,alpha=alpha,symmetric=True,max_iter=100) 
          distances[i,j]=dist
    return distances



def moving_average(series, window_size):

    n = len(series)
    smoothed_series = np.empty(n)
    smoothed_series[:] = np.nan

    for i in range(window_size - 1, n):
        smoothed_series[i] = np.mean(series[i - window_size + 1:i + 1])

    return smoothed_series


def train_val_test_mask(n, train_prop=.6, val_prop=.2):
    """ randomly splits label into train/valid/test splits """
    indices=torch.arange(0,n,1)
    train_num = int(n * train_prop)
    valid_num = int(n * val_prop)
    perm = torch.as_tensor(np.random.permutation(n), dtype=torch.int64)
    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]
    train_mask = torch.zeros(n)
    train_mask[train_indices]=1
    val_mask = torch.zeros(n)
    val_mask[val_indices]=1
    test_mask = torch.zeros(n)
    test_mask[test_indices]=1        
    return train_mask==1, val_mask==1, test_mask==1

def transform_random_split(dataset,train_prop=0.6,val_prop=0.2):
    n=len(dataset.y)
    train_mask,val_mask,test_mask=train_val_test_mask(n,train_prop,val_prop)
    return GraphData(x=dataset.x,y=dataset.y,edge_index=dataset.edge_index,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]
    #print(test_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    
    return data

