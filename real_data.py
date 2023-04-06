import dgl 
from utils import graph_to_adjacency
import torch
import ot

dataset=dgl.data.CiteseerGraphDataset()

g = dataset[0]
features=g.ndata['feat']
label = g.ndata['label']
edges=g.edges()
nodes=g.nodes()

x=ot.dist(x,x)
x=[1]
x=torch.tensor(x,dtype=torch.float)

C=graph_to_adjacency(nodes,edges)
