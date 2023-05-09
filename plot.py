import pylab as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd

#%% Plot initial graph TSNE with connections between nodes

G=torch.load('data/toy_single_train.pt')

X_embedded = TSNE(n_components=2,perplexity=50).fit_transform(G.x)

plt.figure(figsize=(10,8))
edges=G.edge_index
plt.scatter(X_embedded[:,0],X_embedded[:,1],c=G.y,cmap='tab10',vmax=9,alpha=0.5)
for i in range(len(edges[0])):
  plt.plot([X_embedded[edges[0,i],0],X_embedded[edges[1,i],0]],[X_embedded[edges[0,i],1],X_embedded[edges[1,i],1]], alpha=0.01, color='grey')

plt.title('TSNE initial')
plt.show()

#%% Plot latent embedding TSNE with shapes for train/validation

plt.figure(2,figsize=(10,8))

latent=pd.read_csv('results/LTFGW_GCN_complete_graph/Toy_graph_single_seed20_visus.csv')
print(latent)
latent=torch.tensor(latent.values)
latent=latent[:,1:]

X_embedded1 = TSNE(n_components=2,perplexity=40).fit_transform(latent)

edges=G.edge_index

#masks for train and test

mask_val=G.val_mask.numpy()
mask_train=(1-mask_val)==1

plt.scatter(X_embedded1[:,0][G.val_mask],X_embedded1[:,1][G.val_mask],c=G.y[G.val_mask],cmap='tab10',vmax=9,alpha=0.5,marker='o',label='validation') 
plt.scatter(X_embedded1[:,0][mask_train],X_embedded1[:,1][mask_train],c=G.y[mask_train],cmap='tab10',vmax=9,alpha=0.5,marker='v',label='train') 


plt.legend()
plt.title('TSNE - LTFGW_GCN')
