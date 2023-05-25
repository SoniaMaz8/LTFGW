import dgl 
import torch
from torch_geometric.data import Data as GraphData

def Citeseer_data():
  dataset=dgl.data.CiteseerGraphDataset()
  g = dataset[0]
  features=g.ndata['feat']
  edges=g.edges()
  n=len(edges[0])
  edges=torch.cat((edges[0],edges[1]))
  edges=edges.reshape(2,n)
  label = g.ndata['label']
  n_feat=len(features[0])
  G=GraphData(x=features, edge_index=edges,y=label, num_features=n_feat , num_classes=6,train_mask = g.ndata['train_mask'],val_mask=g.ndata['val_mask'],test_mask = g.ndata['test_mask'])
  return(G)


def Cornell():
  edges=torch.load('data/edges_cornell.pt')
  features=torch.load('data/features_cornell.pt')
  labels=torch.load('data/labels_cornell.pt')
  split=torch.load('cornell_split_0.6_0.2_0.npz')
  G=GraphData(x=features, edge_index=edges,y=labels,num_features=len(features[0]),num_classes=5,train_mask=split['train_mask'],val_mask=split['val_mask'],test_mask=split['test_mask'])
  return G