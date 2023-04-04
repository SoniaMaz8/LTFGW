import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from utils import plot_graph, distance_to_template, get_sbm
from torch_geometric.utils import to_networkx
from networkx.generators.community import stochastic_block_model as sbm
from sklearn.manifold import MDS


rng = np.random.RandomState(42)

#toy graph

n = 20 #number of nodes
nc = 3
ratio = np.array([.2, .3, .5])
P = np.array(0.6 * np.eye(3) + 0.02 * np.ones((3, 3)))
C1 = get_sbm(n, nc, ratio, P) 

# get 2d position for nodes
x1 = MDS(dissimilarity='precomputed', random_state=0).fit_transform(1 - C1)

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

feat_C1=[0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
feat_C1=feat_C1+np.random.rand(20)
 
#compute the templates with SBM generator


np.random.seed(42)
N = 10  # number of templates
n_cluster = 2
dataset = []
features_templates = []

#inter-intra cluster probability
p_inter = 0.3
p_intra = 0.6

#templates generation
for i in range(N):
    n_nodes = int(np.random.uniform(low=5, high=7))
    P = p_inter * np.ones((n_cluster, n_cluster))
    np.fill_diagonal(P, p_intra)
    sizes=np.round(n_nodes * np.ones(n_cluster) / n_cluster).astype(np.int32)
    G = sbm(sizes, P, seed=i, directed=False)
    C = nx.to_numpy_array(G)
    m=C.shape[0]
    dataset.append(C) 
    feat=torch.tensor(np.random.randint(1,4,m),dtype=torch.long)
    features_templates.append(feat)

plt.figure(2,figsize=(13,6))
plt.clf()
for i in range(N):
  C = dataset[i]
  x = MDS(dissimilarity='precomputed', random_state=0).fit_transform(C)
  plt.subplot(1, N, i+1)
  plot_graph(x, C)
  plt.title('{}'.format(i+1), fontsize=14)
  plt.axis("off")
plt.show()

#distances to subgraphs computations
nb_neighbours=1
feat_C1=torch.tensor(feat_C1,dtype=torch.long)
feat_C1=feat_C1.long()
dist= distance_to_template(C1,feat_C1,dataset,features_templates,nb_neighbours)

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
