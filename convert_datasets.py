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
