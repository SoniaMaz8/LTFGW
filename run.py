import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from utils import  distance_to_template, get_sbm, plot_graph
from networkx.generators.community import stochastic_block_model as sbm
from sklearn.manifold import MDS
from torch_geometric.utils.convert import from_networkx



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

plt.figure(1, (10, 5))
plt.clf()
plt.subplot(1, 2, 1)
plot_graph(x1, C1, color='C0')
plt.title("Toy graph")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(C1, interpolation='nearest')
plt.title("Adjacency matrix")
plt.axis("off")
plt.show()

#node features

n_feat=3       #dimension of the features
feat_C1=[]   #features
labels=[0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
for i in range(n):               #one hot encoding for the features
   feat=torch.zeros(n_feat)
   feat[labels[i]]=1
  # feat=feat+torch.rand(n_feat)   #noise
   feat_C1.append(feat)
   

feat_C1 = torch.stack(feat_C1)  
feat_C1=feat_C1.type(torch.float64) 


 
#compute the templates with SBM generator



N = 10  # number of templates
n_cluster = 2
dataset = []
features_templates = []

#inter-intra cluster probability
p_inter = 0.3
p_intra = 0.6

#templates generation
for i in range(N):
    n_nodes = torch.randint(5,7,(1,)).item()
    P = p_inter * torch.ones((n_cluster, n_cluster))
    P.fill_diagonal_(p_intra)
    sizes=torch.round(n_nodes * torch.ones(n_cluster) / n_cluster).type(torch.int64)
    G = sbm(sizes, P, seed=i, directed=False)
    C = torch.tensor(nx.to_numpy_array(G))
    m=C.shape[0]
    C=C.type(torch.float64)
    dataset.append(C)
    feat=torch.tensor(torch.randint(0,2,(m,n_feat)),dtype=torch.float64)
    features_templates.append(feat)

plt.figure(2,figsize=(13,6))
plt.clf()
for i in range(N):
  C = dataset[i]
  x = MDS(dissimilarity='precomputed', random_state=0,normalized_stress='auto').fit_transform(C)
  plt.subplot(1, N, i+1)
  plot_graph(x, C)
  plt.title('{}'.format(i+1), fontsize=14)
  plt.axis("off")
plt.show()


#distances to subgraphs computations
nb_neighbours=1
C1=C1.type(torch.float64) 

dist= distance_to_template(C1,feat_C1,dataset,features_templates,nb_neighbours,n_feat,0.5)


plt.figure(3)
plt.imshow(dist)
plt.title('distance matrix')
plt.xlabel('templates')
plt.ylabel('nodes')
plt.show()

#visualisation of the embeddings along the 2 first dimensions
plt.figure(4)
plt.scatter(dist[:4,0],dist[:4,1],color='r',label='class 1')
plt.scatter(dist[4:10,0],dist[4:10,1],color='b',label='class 2')
plt.scatter(dist[10:20,0],dist[10:20,1],color='g',label='class 3')
plt.title('visualisation of the embeddings along the 2 first dimensions')
plt.legend()
plt.show()
