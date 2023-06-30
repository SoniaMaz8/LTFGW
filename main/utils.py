import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import to_networkx
import os
from data.convert_datasets import Citeseer_data
from GNN.architectures import *


def get_model(
        model_name:str,
        n_classes:int,
        n_features:int,
        n_templates:int,
        n_templates_nodes:int,
        hidden_layer:int,
        dropout:float,
        shortest_path:bool,
        k:int,
        mean_init:float,
        std_init:float,
        log:bool,
        alpha0,
        train_node_weights:bool,
        skip_connection:bool,
        template_sizes,
        n_hidden_layer:int,
        reg):

    
    """"
    Function to get the model.

    Parameters
    ----------

    model_name: str
      Name of the model.
    n_classes: int
      Number of classes for node classification.
    n_features: int
      Number of features for each node.
    n_nodes: int
      Number of nodes for the LTFGW templates.
    mean_init: float
      Mean of the normal distribution for the initialisation of the LTFGW templates.
    std_init: float
      Standard deviation of the normal distribution for the initialisation of the LTFGW templates.
    template_sizes: if None, all template have the same number of nodes. 
        Else, list of the number of nodes of the templates. 
    
    """
    if not model_name in ['LTFGW_GCN','MLP','GCN','LTFGW_MLP','ChebNet','GCN_JK','LTFGW_MLP_semirelaxed', 'LTFGW_MLP_dropout', 'LTFGW_MLP_dropout_relu', 'LTFGW_MLP_dropout_relu_one_node']:
        raise ValueError(
            'The model is not supported.')
    
    elif model_name == 'LTFGW_GCN':
        model = LTFGW_GCN(n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log,alpha0,train_node_weights, skip_connection ,template_sizes)

    elif model_name == 'MLP':
        model = MLP(n_hidden_layer, hidden_layer, dropout, n_classes, n_features)

    elif model_name == 'GCN':
        model = GCN( n_classes, n_features, hidden_layer, n_hidden_layer, dropout)

    elif model_name == 'LTFGW_MLP':
        model = LTFGW_MLP(n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log,alpha0,train_node_weights, skip_connection ,template_sizes)

    elif model_name == 'ChebNet':
        model = ChebNet(dropout, n_classes, n_features)

    elif model_name == 'GCN_JK':
        model = GCN_JK(n_classes, n_features, hidden_layer)

    elif model_name == 'LTFGW_MLP_semirelaxed':
        model = LTFGW_MLP_semirelaxed(n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log,alpha0,train_node_weights, skip_connection,template_sizes, reg)
        

    elif model_name == 'LTFGW_MLP_dropout':
        model = LTFGW_MLP_dropout(n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log,alpha0,train_node_weights, skip_connection ,template_sizes)
        
    
    elif model_name == 'LTFGW_MLP_dropout_relu':
        model = LTFGW_MLP_dropout_relu( n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log,alpha0,train_node_weights, skip_connection ,template_sizes)          

    return model


def get_dataset(dataset_name):
    """
    Function that returns the dataset and the number of classes for a given
    dataset name

    """

    if dataset_name == 'Citeseer':
        dataset = Citeseer_data()
        n_classes = 6
        n_features = dataset.num_features
        test_graph = False
        graph_type = 'single_graph'

    elif dataset_name == 'Toy_graph_single':
        dataset_train = torch.load('data/toy_single_train.pt')
        dataset_test = torch.load('data/toy_single_test.pt')
        dataset = [dataset_train, dataset_test]
        n_classes = 3
        n_features = dataset_train.num_features
        test_graph = True
        graph_type = 'single_graph'

    elif dataset_name == 'Toy_graph_multi':
        dataset = torch.load('data/toy_multi_graph.pt')
        n_classes = 3
        n_features = dataset[0].num_features
        test_graph = None
        graph_type = 'multi_graph'

    elif dataset_name == 'mutag':
        dataset = torch.load('data/mutag.pt')
        n_classes = 7
        n_features = dataset[0].num_features
        test_graph = None
        graph_type = 'multi_graph'

    elif dataset_name == 'cornell':
        dataset = torch.load('data/cornell_undirected.pt')
        n_classes = 5
        n_features = dataset.num_features
        test_graph = False
        graph_type = 'single_graph'

    elif dataset_name == 'cornell_directed':
        dataset = torch.load('data/cornell.pt')
        n_classes = 5
        n_features = dataset.num_features
        test_graph = False
        graph_type = 'single_graph'    

    elif dataset_name == 'chameleon':
        dataset = torch.load('data/chameleon_undirected.pt')
        n_classes = 5
        n_features = dataset.num_features
        test_graph = False
        graph_type = 'single_graph'

    elif dataset_name == 'wisconsin':
        dataset = torch.load('data/winsconsin_undirected.pt')
        n_classes = 5
        n_features = dataset.num_features
        test_graph = False
        graph_type = 'single_graph'

    elif dataset_name == 'anti_sbm':
        dataset = torch.load('data/anti_sbm.pt')
        n_classes = 3
        n_features = 3
        test_graph = False
        graph_type = 'single_graph'    

    mean = float(torch.mean(dataset.x).item())
    std = float(torch.std(dataset.x).item())

    return dataset, n_classes, n_features, test_graph, graph_type, mean, std


def get_filenames(
        dataset_name,
        method,
        lr,
        n_templates,
        n_nodes,
        alpha0,
        k,
        dropout,
        wd,
        hidden_layer,
        scheduler,
        seed=None,
        template_sizes=None,
        log=False):
    
    hl=hidden_layer
    n_temp=n_templates
    
    performance_dir=os.path.join('results',method,dataset_name,str(
                seed.item()),'performances')
    best_model_dir=os.path.join('results',method,dataset_name,str(
                seed.item()),'best_model')
    current_model_dir= os.path.join('results',method,dataset_name,str(
                seed.item()),'current_model')
    visus_dir= os.path.join('results',method,dataset_name,str(
                seed.item()),'visus')
    templates_dir= os.path.join('results',method,dataset_name,str(
                seed.item()),'templates')
    alphas_dir= os.path.join('results',method,dataset_name,str(
                seed.item()),'alphas')
    if not os.path.isdir(performance_dir):
        os.makedirs(performance_dir)
    if not os.path.isdir(best_model_dir):
        os.makedirs(best_model_dir)
    if not os.path.isdir(current_model_dir):
        os.makedirs(current_model_dir)
    if not os.path.isdir(visus_dir):
        os.makedirs(visus_dir)
    if not os.path.isdir(templates_dir):
        os.makedirs(templates_dir)
    if not os.path.isdir(alphas_dir):
        os.makedirs(alphas_dir)

    if seed is None:
        filename_save = os.path.join(
            'results', method, "{}.pkl".format(dataset_name))
        filename_best_model = os.path.join(
            'results', method, "{}_best_valid.pkl".format(dataset_name))
        filename_visus = os.path.join(
            'results', method, "{}_visus.pkl".format(dataset_name))

    elif template_sizes==None:
        filename_save = os.path.join(
            performance_dir,
            "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}.pkl".format(
                lr,
                n_temp,
                n_nodes,
                alpha0,
                k,
                dropout,
                wd,
                hl,
                scheduler,
                log))
        filename_best_model = os.path.join(
            best_model_dir,
            "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}.pt".format(
                lr,
                n_temp,
                n_nodes,
                alpha0,
                k,
                dropout,
                wd,
                hl,
                scheduler,
                log))
        filename_visus = os.path.join(
            visus_dir, "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}.pkl".format(
                lr, n_temp, n_nodes, alpha0, k, dropout, wd, hl, scheduler,log))
        filename_current_model = os.path.join(
            current_model_dir,
            "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}.pt".format(
                lr,
                n_temp,
                n_nodes,
                alpha0,
                k,
                dropout,
                wd,
                hl,
                scheduler,
                log))
        filename_templates = os.path.join(
                templates_dir, "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}.pkl".format(
                lr, n_temp, n_nodes, alpha0, k, dropout, wd, hl, scheduler,log))
        filename_alphas = os.path.join(
            alphas_dir,
            "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}.pkl".format(
                lr, n_temp, n_nodes, alpha0, k, dropout, wd, hl, scheduler,log))
        
    else:
        filename_save = os.path.join(
            performance_dir,
            "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}_tempsizes.pkl".format(
                lr,
                n_temp,
                n_nodes,
                alpha0,
                k,
                dropout,
                wd,
                hl,
                scheduler,
                log))
        filename_best_model = os.path.join(
            best_model_dir,
            "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}_tempsizes.pt".format(
                lr,
                n_temp,
                n_nodes,
                alpha0,
                k,
                dropout,
                wd,
                hl,
                scheduler,
                log))
        filename_visus = os.path.join(
            visus_dir, "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}_tempsizes.pkl".format(
                lr, n_temp, n_nodes, alpha0, k, dropout, wd, hl, scheduler,log))
        filename_current_model = os.path.join(
            current_model_dir,
            "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}_tempsizes.pt".format(
                lr,
                n_temp,
                n_nodes,
                alpha0,
                k,
                dropout,
                wd,
                hl,
                scheduler,
                log))
        filename_templates = os.path.join(
            templates_dir, "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}_tempsizes.pkl".format(
                lr, n_temp, n_nodes, alpha0, k, dropout, wd, hl, scheduler,log))
        filename_alphas = os.path.join(
            alphas_dir, "lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler_{}_log{}_tempsizes.pkl".format(
                lr, n_temp, n_nodes, alpha0, k, dropout, wd, hl, scheduler,log))

    return filename_save, filename_best_model, filename_visus, filename_current_model, filename_templates, filename_alphas


def visualize_graph(G, color='b'):
    """"
    visualisation of a torch.geometric graph
    """
    plt.figure(figsize=(7, 7))
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
                    plt.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]],
                             alpha=0.2, color='k')
            else:  # connection intensity proportional to C[i,j]
                plt.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]],
                         alpha=C[i, j], color='k')

    plt.scatter(x[:, 0], x[:, 1], c=color, s=s, zorder=10,
                edgecolors='k', cmap='tab10', vmax=9)


def get_sbm(n, nc, ratio, P):
    nbpc = torch.round(n * ratio).type(torch.int64)
    n = torch.sum(nbpc).item()
    C = torch.zeros(n, n)
    for c1 in range(nc):
        for c2 in range(c1 + 1):
            if c1 == c2:
                for i in range(torch.sum(nbpc[:c1]), torch.sum(nbpc[:c1 + 1])):
                    for j in range(torch.sum(nbpc[:c2]), i):
                        if torch.rand(1) <= P[c1, c2]:
                            C[i, j] = 1
            else:
                for i in range(torch.sum(nbpc[:c1]), torch.sum(nbpc[:c1 + 1])):
                    for j in range(
                            torch.sum(nbpc[:c2]), torch.sum(nbpc[:c2 + 1])):
                        if torch.rand(1) <= P[c1, c2]:
                            C[i, j] = 1
    return C + C.T


def moving_average(series, window_size):

    n = len(series)
    smoothed_series = np.empty(n)
    smoothed_series[:] = np.nan

    for i in range(window_size - 1, n):
        smoothed_series[i] = np.mean(series[i - window_size + 1:i + 1])

    return smoothed_series


def train_val_test_mask(n, train_prop=.6, val_prop=.2):
    """ randomly splits label into train/valid/test splits """
    indices = torch.arange(0, n, 1)
    train_num = int(n * train_prop)
    valid_num = int(n * val_prop)
    perm = torch.as_tensor(np.random.permutation(n), dtype=torch.int64)
    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]
    train_mask = torch.zeros(n)
    train_mask[train_indices] = 1
    val_mask = torch.zeros(n)
    val_mask[val_indices] = 1
    test_mask = torch.zeros(n)
    test_mask[test_indices] = 1
    return train_mask == 1, val_mask == 1, test_mask == 1


def transform_random_split(dataset, train_prop=0.6, val_prop=0.2):
    n = len(dataset.y)
    train_mask, val_mask, test_mask = train_val_test_mask(
        n, train_prop, val_prop)
    return GraphData(
        x=dataset.x,
        y=dataset.y,
        edge_index=dataset.edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask)


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_planetoid_splits(
        data,
        num_classes,
        percls_trn=20,
        val_lb=500,
        seed=12134):
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(
                rnd_state.choice(
                    class_idx,
                    percls_trn,
                    replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]
    # print(test_idx)

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    return data
