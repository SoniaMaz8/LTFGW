from dgl.data import CiteseerGraphDataset
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import time
import numpy as np
import torch.nn.functional as F
from trainers import test,test_multigraph
from GNN.architectures import LTFGW_GCN,GCN,MLP
from GNN.utils import get_dataset


dataset=torch.load('data/toy_multi_graph.pt')

seed=23
torch.manual_seed(seed)

generator = torch.Generator().manual_seed(seed)
train_dataset,val_dataset,test_dataset=torch.utils.data.random_split(dataset,[600,200,200],generator=generator)

n_classes=3


model=LTFGW_GCN(n_classes=n_classes,n_features=3, n_templates=3,n_templates_nodes=3)

checkpoint = torch.load('results/LTFGW_GCN_multi_graph/Toy_graph_multi_seed23_best_valid.pkl')
model.load_state_dict(checkpoint['model_state_dict'])

print(test_multigraph(model,dataset))