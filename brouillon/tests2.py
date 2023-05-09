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


def multi_graph(n=1000,p_intra=0.6,p_inter=0.06,min_n_nodes=5,max_n_nodes=10):
   """"
   Function that constructs a multi graph of sbms 

   n_train: number of training graphs
   n_val: number of validation graphs
   n_test: number of test graphs
   p_inter: inter probability for the sbms
   p_intra: intra probability for the sbms
   min_n_nodes: minimal number of nodes in the graphs
   max_n_nodes: maximal number of nodes in the graphs
   """

   graphs=[]
   n_nodes=torch.randint(low=min_n_nodes,high=max_n_nodes,size=[n])
   for i in range(n):
      n = n_nodes[i]  #number of nodes
      nc = 3   #number of clusters
      ratio=torch.rand(3)
      ratio=F.normalize(ratio,p=1,dim=0)
      P = p_intra * torch.eye(3) + p_inter * torch.ones(3, 3)
      C = get_sbm(n, nc, ratio, P)
      
      #in the case when C and 1 differ from 1, because of approximations (round)

      n=C.shape[0]

      #labels according to the random ratio

      ratio=torch.round(ratio*n)
      if torch.sum(ratio)<n:
         ratio[torch.randint(0,3,size=[1])]+=1
      elif torch.sum(ratio)>n:
         ratio[torch.argmax(ratio)]-=1

      labels1=torch.zeros(int(ratio[0].item()))
      labels2=torch.ones(int(ratio[1].item()))
      labels3=torch.ones(int(ratio[2].item()))*2
      labels=torch.hstack([labels1,labels2,labels3])
      labels=torch.Tensor(labels).type(torch.LongTensor)

      #features as one hot encoding + noise of the labels

      n_feat=3
      features=[]
      for j in range(n):               
         feat=torch.zeros(n_feat)
         feat[labels[j]]=1
         features.append(feat)

      features = torch.stack(features) 
      features=features+torch.normal(torch.zeros(n,n_feat),torch.ones(n,n_feat))  #noise on the features

      #build the graphs

      G=adjacency_to_graph(C,features)
      G=GraphData(x=features, edge_index=G.edge_index,y=labels, num_features=n_feat , num_classes=3, num_nodes=n)
      
      graphs.append(G)
       
   return graphs


graphs=multi_graph()


torch.save(graphs,'data/toy_multi_graph.pt')

