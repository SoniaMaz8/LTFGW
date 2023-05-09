#%%

import torch
import matplotlib.pyplot as plt
from GNN.utils import  distance_to_template, get_sbm, subgraph,plot_graph, adjacency_to_graph
from networkx.generators.community import stochastic_block_model as sbm
from sklearn.manifold import MDS
from torch_geometric.data import Data as GraphData
from torch_geometric.transforms import RandomNodeSplit
from sklearn.manifold import TSNE
import torch.nn.functional as F

#%% Train toy graph

#toy graph

n = 1000   #number of nodes
nc = 3   #number of clusters
ratio = torch.tensor([.333, .333, .334])
P = 0.01 * torch.eye(3) + 0.005 * torch.ones(3, 3)
C1 = get_sbm(n, nc, ratio, P)
print(torch.sum(C1)/1000)

#Node features

n_feat=3       #dimension of the features
feat_C1=[]   #features
labels1=torch.zeros(333)
labels2=torch.ones(333)
labels3=torch.ones(334)*2
labels=torch.hstack([labels1,labels2,labels3])
labels=torch.Tensor(labels).type(torch.LongTensor)

for i in range(n):               #one hot encoding for the features
   feat=torch.zeros(n_feat)
   feat[labels[i]]=1
   feat_C1.append(feat)
   
feat_C1 = torch.stack(feat_C1) 
feat_C1=feat_C1+torch.normal(torch.zeros(n,n_feat),torch.ones(n,n_feat))  #noise on the features

G=adjacency_to_graph(C1,feat_C1)

G1=GraphData(x=feat_C1, edge_index=G.edge_index,y=labels, num_features=n_feat , num_classes=3)

transform=RandomNodeSplit(num_val=500,num_test=0)  #split into train set and validation set
G1=transform(G1)
#torch.save(G1,'data/toy_single_train.pt')



#%% Test toy graph


#toy graph

n = 1000   #number of nodes
nc = 3   #number of clusters
ratio = torch.tensor([.333, .333, .334])
P = 0.01 * torch.eye(3) + 0.005 * torch.ones(3, 3)
C2 = get_sbm(n, nc, ratio, P)

#Node features

n_feat=3       #dimension of the features
feat_C2=[]   #features
labels1=torch.zeros(333)
labels2=torch.ones(333)
labels3=torch.ones(334)*2
labels=torch.hstack([labels1,labels2,labels3])
labels=torch.Tensor(labels).type(torch.LongTensor)

for i in range(n):               #one hot encoding for the features
   feat=torch.zeros(n_feat)
   feat[labels[i]]=1   #noise
   feat_C2.append(feat)
   

feat_C2 = torch.stack(feat_C2)  
feat_C2=feat_C2+torch.normal(torch.zeros(n,n_feat),torch.ones(n,n_feat))

G=adjacency_to_graph(C2,feat_C2)

G2=GraphData(x=feat_C2, edge_index=G.edge_index,y=labels, num_features=n_feat , num_classes=3)

#torch.save(G2,'data/toy_single_test.pt')

#plt.figure(1, (10, 5))
#plt.subplot(1,2,1)
#plt.imshow(C1, interpolation='nearest')
#plt.title("Adjacency matrix train")
#plt.axis("off")
#plt.subplot(1,2,2)
#plt.imshow(C2, interpolation='nearest')
#plt.title("Adjacency matrix test")
#plt.axis("off")
#plt.savefig('toy_adjacency.png')
#plt.show()


#%% Multi graph

def multi_graph(n=1000,p_inter=0.5,p_intra=0.05,min_n_nodes=5,max_n_nodes=10):
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