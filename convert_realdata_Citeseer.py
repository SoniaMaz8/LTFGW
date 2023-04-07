import dgl 
from utils import graph_to_adjacency
import torch
import numpy as np


dataset=dgl.data.CiteseerGraphDataset()

g = dataset[0]
features=g.ndata['feat']
label = g.ndata['label']
edges=g.edges()
nodes=g.nodes()
n_feat=len(features[0])
n=len(edges[0])
edges=torch.cat((edges[0],edges[1]))
edges=edges.reshape(2,n)
n_nodes=len(nodes)

C1=graph_to_adjacency(n_nodes,edges)

torch.save(C1,'C_Citeseer.pt')
torch.save(edges,'edges_Citeseer.pt')
torch.save(n_nodes,'n_nodes_Citeseer.pt')
torch.save(nodes,'nodes_Citeseer.pt')
torch.save(label,'labels_Citeseer.pt')
torch.save(features,'features_Citeseer.pt')
