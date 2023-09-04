#%%
import torch
from torch_geometric.data import Data as GraphData
from main.utils import *
from main.trainers import *
from GNN.architectures import *

#%%

n_nodes=200
n_classes=3

classes=torch.randint(n_classes,(n_nodes,))


p_intra=0.001
p_inter=0.1
P = p_inter * torch.ones(n_classes, n_classes) - (p_inter-p_intra)*torch.eye(n_classes)


features=torch.zeros(n_nodes,n_classes)

for i in range(len(classes)):
    features[i][classes[i]]=1

noise=torch.rand((n_nodes,n_classes))*6 -3 

features=features+noise

edge_index_1=[]
edge_index_2=[]

for i in range(n_nodes):
    for j in range(i+1,n_nodes):
        if torch.rand(1)<P[classes[i],classes[j]]:
            edge_index_1.append(i)
            edge_index_1.append(j)
            edge_index_2.append(j)
            edge_index_2.append(i)


edge_index_1=torch.tensor(edge_index_1)
edge_index_2=torch.tensor(edge_index_2)
edge_index=torch.vstack([edge_index_1,edge_index_2])
graph=GraphData(x=features,y=classes,edge_index=edge_index)

plt.figure(figsize=(7, 7))
plt.xticks([])
plt.yticks([])
G = to_networkx(graph, to_undirected=True)


#nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
#                 node_color=graph.y, cmap='tab10',alpha=0.5, node_size=100)
#plt.show()

torch.save(graph,'data/anti_sbm1.pt')


# %%
