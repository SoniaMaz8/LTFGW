import dgl
import torch
import torch.nn as nn
import pylab as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from dgl.data import CiteseerGraphDataset
from trainers import train_epoch, test
from data.convert_datasets import Citeseer_data
from sklearn.manifold import TSNE
from GNN.utils import adjacency_to_graph,get_sbm
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader
from GNN.architectures import GCN

dataset_=torch.load('data/toy_multi_graph.pt')

dataset=DataLoader(dataset_,batch_size=100,shuffle=True)
model=GCN(n_classes=3,n_features=3)

for data in dataset:
    print(data)
    out,_=model(data.x,data.edge_index)
 #   print(out.shape)
    pred = out.argmax(dim=1)
    print(pred)

