import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from utils import  distance_to_template, get_sbm, subgraph,plot_graph, adjacency_to_graph
from networkx.generators.community import stochastic_block_model as sbm
from sklearn.manifold import MDS
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data as GraphData
from torch_geometric.transforms import RandomNodeSplit

#%% Train toy graph

#rng = np.random.RandomState(42)
torch.manual_seed(42)

#toy graph

n = 20 #number of nodes
nc = 3
ratio = torch.tensor([.2, .3, .5])
P = 0.6 * torch.eye(3) + 0.02 * torch.ones(3, 3)
C1 = get_sbm(n, nc, ratio, P)

# get 2d position for nodes
x1 = MDS(dissimilarity='precomputed', random_state=2,normalized_stress='auto').fit_transform(1-C1)

#plot the graph

plt.figure(1, (10, 5))
plt.clf()
plt.subplot(1, 2, 1)
plot_graph(x1, C1, color='C0')
plt.title("Train toy graph")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(C1, interpolation='nearest')
plt.title("Adjacency matrix")
plt.axis("off")
plt.show()

#node features

n_feat=3       #dimension of the features
feat_C1=[]   #features
labels=torch.tensor([0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2])
for i in range(n):               #one hot encoding for the features
   feat=torch.zeros(n_feat)
   feat[labels[i]]=1
  # feat=feat+torch.rand(n_feat)   #noise
   feat_C1.append(feat)
   

feat_C1 = torch.stack(feat_C1)  

G=adjacency_to_graph(C1,feat_C1)

G1=GraphData(x=feat_C1, edge_index=G.edge_index,y=labels, num_features=n_feat , num_classes=3)

transform=RandomNodeSplit(num_train_per_class=2,num_val=0,num_test=12)  #split into test set,train set
G1=transform(G1)
G1=torch.save(G1,'toy_graph1.pt')

#%% Test toy graph


torch.manual_seed(32)

#toy graph

n = 20 #number of nodes
nc = 3
ratio = torch.tensor([.3, .3, .4])
P = 0.6 * torch.eye(3) + 0.02 * torch.ones(3, 3)
C2 = get_sbm(n, nc, ratio, P)

# get 2d position for nodes
x2 = MDS(dissimilarity='precomputed', random_state=2,normalized_stress='auto').fit_transform(1-C1)

#plot the graph

plt.figure(2, (10, 5))
plt.clf()
plt.subplot(1, 2, 1)
plot_graph(x2, C2, color='C0')
plt.title("Test toy graph")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(C2, interpolation='nearest')
plt.title("Adjacency matrix")
plt.axis("off")
plt.show()

#node features

n_feat=3       #dimension of the features
feat_C2=[]   #features
labels=torch.tensor([0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2])
for i in range(n):               #one hot encoding for the features
   feat=torch.zeros(n_feat)
   feat[labels[i]]=1
  # feat=feat+torch.rand(n_feat)   #noise
   feat_C2.append(feat)
   

feat_C2 = torch.stack(feat_C2)  

G=adjacency_to_graph(C2,feat_C2)

G2=GraphData(x=feat_C2, edge_index=G.edge_index,y=labels, num_features=n_feat , num_classes=3)

transform=RandomNodeSplit(num_train_per_class=2,num_val=0,num_test=12)  #split into test set,train set
G2=transform(G2)
G2=torch.save(G2,'toy_graph2.pt')





