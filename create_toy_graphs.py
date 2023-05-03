import torch
import matplotlib.pyplot as plt
from utils import  distance_to_template, get_sbm, subgraph,plot_graph, adjacency_to_graph
from networkx.generators.community import stochastic_block_model as sbm
from sklearn.manifold import MDS
from torch_geometric.data import Data as GraphData
from torch_geometric.transforms import RandomNodeSplit
from sklearn.manifold import TSNE

#%% Train toy graph

#toy graph

n = 1000   #number of nodes
nc = 3   #number of clusters
ratio = torch.tensor([.3, .3, .3])
P = 0.05 * torch.eye(3) + 0.01 * torch.ones(3, 3)
C1 = get_sbm(n, nc, ratio, P)

#Node features

n_feat=3       #dimension of the features
feat_C1=[]   #features
labels1=torch.zeros(333)
labels2=torch.ones(333)
labels3=torch.ones(334)*2
labels=torch.hstack([labels1,labels2,labels3])
labels=torch.Tensor.int(labels)
for i in range(n):               #one hot encoding for the features
   feat=torch.zeros(n_feat)
   feat[labels[i]]=1
   feat=feat+2*torch.rand(n_feat)   #noise
   feat_C1.append(feat)
   

feat_C1 = torch.stack(feat_C1)  

G=adjacency_to_graph(C1,feat_C1)

G1=GraphData(x=feat_C1, edge_index=G.edge_index,y=labels, num_features=n_feat , num_classes=3)

transform=RandomNodeSplit(num_val=500,num_test=0)  #split into test set,train set
G1=transform(G1)
torch.save(G1,'data/toy_single_train.pt')



#%% Test toy graph


#toy graph

n = 1000   #number of nodes
nc = 3   #number of clusters
ratio = torch.tensor([.3, .3, .3])
P = 0.05 * torch.eye(3) + 0.01 * torch.ones(3, 3)
C2 = get_sbm(n, nc, ratio, P)

#Node features

n_feat=3       #dimension of the features
feat_C2=[]   #features
labels1=torch.zeros(333)
labels2=torch.ones(333)
labels3=torch.ones(334)*2
labels=torch.hstack([labels1,labels2,labels3])
labels=torch.Tensor.int(labels)
for i in range(n):               #one hot encoding for the features
   feat=torch.zeros(n_feat)
   feat[labels[i]]=1
   feat=feat+2*torch.rand(n_feat)   #noise
   feat_C2.append(feat)
   

feat_C2 = torch.stack(feat_C2)  

G=adjacency_to_graph(C2,feat_C2)

G2=GraphData(x=feat_C2, edge_index=G.edge_index,y=labels, num_features=n_feat , num_classes=3)

torch.save(G2,'data/toy_single_test.pt')

plt.figure(1, (10, 5))
plt.subplot(1,2,1)
plt.imshow(C1, interpolation='nearest')
plt.title("Adjacency matrix train")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(C2, interpolation='nearest')
plt.title("Adjacency matrix test")
plt.axis("off")
plt.show()
