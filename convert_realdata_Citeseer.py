import dgl 
from utils import graph_to_adjacency, distance_to_template,adjacency_to_graph
import torch
import numpy as np
import networkx as nx
from networkx.generators.community import stochastic_block_model as sbm
import matplotlib.pyplot as plt


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


#templates generation

#np.random.seed(42)
#N = 30  # number of templates
#n_cluster = 2
#dataset = []
#features_templates = []

#inter-intra cluster probability
#p_inter = 0.3
#p_intra = 0.6

#templates generation
#for i in range(N):
#    n_nodes = int(np.random.uniform(low=5, high=7))
#    P = p_inter * np.ones((n_cluster, n_cluster))
#    np.fill_diagonal(P, p_intra)
#    sizes=np.round(n_nodes * np.ones(n_cluster) / n_cluster).astype(np.int32)
#    G = sbm(sizes, P, seed=i, directed=False)
#    C = nx.to_numpy_array(G)
#    m=C.shape[0]
#    dataset.append(C) 
#    feat=torch.tensor(np.random.randint(0,2,(m,n_feat)),dtype=torch.long)
#    features_templates.append(feat)





#nb_neighbours=1

#C1=torch.tensor(C1,dtype=torch.float)
#C1=C1.type(torch.float)
#features=torch.tensor(features)
#features=features.long()



dist= distance_to_template(C1,features,dataset,features_templates,nb_neighbours,n_feat,0.5)


np.save('distance2',dist)