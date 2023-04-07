import dgl 
from utils import graph_to_adjacency
import torch
import numpy as np

#Nodes mean scientific publications and edges mean citation relationships. Each node has a predefined feature with 3703 dimensions. 

dataset=dgl.data.CiteseerGraphDataset()

#extract the useful data
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

#compute the adjacency matrix
C1=graph_to_adjacency(n_nodes,edges)


#save the data
torch.save(C1,'C_Citeseer.pt')
torch.save(edges,'edges_Citeseer.pt')
torch.save(n_nodes,'n_nodes_Citeseer.pt')
torch.save(nodes,'nodes_Citeseer.pt')
torch.save(label,'labels_Citeseer.pt')
torch.save(features,'features_Citeseer.pt')
torch.save(G,'graph_Citeseer.pt')
