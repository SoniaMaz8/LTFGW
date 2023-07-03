from sklearn.model_selection import ParameterGrid

# %%

import torch_geometric
from torch_geometric.loader import NeighborLoader, DataLoader, ClusterData, ClusterLoader, DataLoader
from GNN.architectures import *

from main.utils import *
from main.trainers import *
import os
import torch
import numpy as np
import torch_geometric.transforms as T
import argparse

torch_geometric.typing.WITH_PYG_LIB = False

# %%Parameters to set



param_grid={'lr':[0.1,0.01,0.5,0.05,0.001,0.005,0.0001,0.0005],'wd':[0,0.0005,0.005,0.05,0.5],'drop':[0,0.2,0.4]}
seed=torch.tensor([21])

dataset_name='anti_sbm'

method = 'MLP'

n_hidden_layer=0

hidden_layer=3

model_name='MLP'

n_templates=0
n_template_nodes=0
shortest_path=True
k=0
log=False
alpha0=None
train_node_weights=False
skip_connection=False
template_sizes=None
reg=False
scheduler=False
nepochs=1000
save=True
hidden_layer=3


# %%Training and testing

criterion = torch.nn.CrossEntropyLoss()

# load dataset
dataset, n_classes, n_features, test_graph, graph_type, mean_init, std_init = get_dataset(dataset_name)

n_nodes = len(dataset.x)

all_params=[]
all_perf=[]

for params in ParameterGrid(param_grid):

    dropout=params['drop']
    lr=params['lr']
    wd=params['wd']

    torch.manual_seed(seed)

    model = get_model( model_name,
        n_classes,
        n_features,
        n_templates,
        n_template_nodes,
        hidden_layer,
        dropout,
        shortest_path,
        k,
        mean_init,
        std_init,
        log,
        alpha0,
        train_node_weights,
        skip_connection,
        template_sizes,
        n_hidden_layer,
        reg)
     
    if alpha0 is None:
        filename_save, filename_best_model, filename_visus, filename_current_model, filename_templates, filename_alpha = get_filenames(dataset_name,
        method,
        lr,
        n_templates,
        n_template_nodes,
        alpha0,
        k,
        dropout,
        wd,
        hidden_layer,
        scheduler,
        seed,
        template_sizes,
        log)
    else:
        filename_save, filename_best_model, filename_visus, filename_current_model, filename_templates, filename_alpha = get_filenames(dataset_name,
        method,
        lr,
        n_templates,
        n_nodes,
        alpha0.item(),
        k,
        dropout,
        wd,
        hidden_layer,
        scheduler,
        seed,
        template_sizes,
        log)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd)

    if graph_type == 'single_graph':
        percls_trn = int(round(0.6 * len(dataset.y) / n_classes))
        val_lb = int(round(0.2 * len(dataset.y)))
        dataset = random_planetoid_splits(
            dataset,
            n_classes,
            percls_trn=percls_trn,
            val_lb=val_lb,
            seed=seed)
        loader = NeighborLoader(dataset,
                                num_neighbors=[-1],
                                input_nodes=dataset.train_mask,
                                directed=False,
                                batch_size=torch.sum(dataset.train_mask).item()
                                )
        loader_val = NeighborLoader(dataset,
                                    num_neighbors=[-1],
                                    input_nodes=dataset.val_mask,
                                    directed=False,
                                    batch_size=torch.sum(dataset.val_mask).item()
                                    )
        dataset_test = dataset
        train(criterion,optimizer,loader,loader_val,model,filename_save,filename_best_model,filename_visus,filename_templates,filename_alpha,filename_current_model,save,scheduler, template_sizes,nepochs,model_name)



    elif graph_type == 'mini_batch':
        percls_trn = int(round(0.6 * len(dataset.y) / n_classes))
        val_lb = int(round(0.2 * len(dataset.y)))
        dataset = random_planetoid_splits(
            dataset,
            n_classes,
            percls_trn=percls_trn,
            val_lb=val_lb,
            seed=seed)

        train_minibatch(
            model,
            dataset,
            nepochs,
            criterion,
            optimizer,
            save ,
            filename_save,
            filename_best_model,
            filename_visus,
            loader,
            loader_val,
            filename_current_model)

    checkpoint = torch.load(filename_best_model)
    model.load_state_dict(checkpoint['model_state_dict'])

    if graph_type == 'multi_graph':
        test_acc = test_multigraph(model, dataset_test)


    elif graph_type == 'single_graph':
        test_acc = test(model, dataset_test, test_graph)


    elif graph_type == 'mini_batch':
        test_acc = test(model, dataset_test, test_graph)

    print(params)
    print(test_acc)

    all_params.append(params)
    all_perf.append(test_acc)

pd.DataFrame(all_params).to_csv('all_params.csv')
pd.DataFrame(all_perf).to_csv('all_perf.csv')






