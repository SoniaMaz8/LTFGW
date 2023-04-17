import dgl 
from utils import graph_to_adjacency, distance_to_template,adjacency_to_graph
import torch
import numpy as np
import networkx as nx
from networkx.generators.community import stochastic_block_model as sbm
import matplotlib.pyplot as plt
from torch_geometric.data import Data as GraphData
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import Entities
import pandas as pd
from torch_geometric.datasets import TUDataset

torch.manual_seed(12345)

transform=RandomNodeSplit(num_train_per_class=3,num_val=0,num_test=8)

dataset = TUDataset(root='data/TUDataset', name='MUTAG',transform=transform)


dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]


torch.save(train_dataset,'train_mutag.pt')
torch.save(test_dataset,'test_mutag.pt')