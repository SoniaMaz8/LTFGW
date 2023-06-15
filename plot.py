#%%

import pylab as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd
from GNN.utils import moving_average


#%% Plot initial graph TSNE with connections between nodes

G=torch.load('data/mutag.pt')
G=G[0]
print(G.x)

X_embedded = TSNE(n_components=7,perplexity=5).fit_transform(G.x)
edges=G.edge_index

plt.figure(figsize=(10,8))
plt.scatter(X_embedded[:,0],X_embedded[:,1],c=G.y,cmap='tab10',vmax=9,alpha=0.5)

#plot connections 
for i in range(len(edges[0])):
  plt.plot([X_embedded[edges[0,i],0],X_embedded[edges[1,i],0]],[X_embedded[edges[0,i],1],X_embedded[edges[1,i],1]], alpha=0.01, color='grey')

plt.title('TSNE initial')
plt.show()

#%% Plot latent embedding TSNE with shapes for train/validation

mutag=torch.load('data/mutag.pt')
generator = torch.Generator().manual_seed(20)
train_dataset,val_dataset,test_dataset=torch.utils.data.random_split(mutag,[112,38,38],generator=generator)
G=torch.load('g.pt')

plt.figure(2,figsize=(10,8))

latent=pd.read_pickle('results/LTFGW_GCN_multi_graph/mutag_seed20_visus.pkl')
latent=torch.tensor(latent.values)
latent=latent[:,1:]


X_embedded1 = TSNE(n_components=2,perplexity=30).fit_transform(latent)

edges=G.edge_index

#masks for train and test

#mask_val=G.val_mask.numpy()
#mask_train=(1-mask_val)==1

print(len(G.y))

plt.scatter(X_embedded1[:,0],X_embedded1[:,1],c=G.y,cmap='tab10',vmax=9,alpha=0.5,marker='o') 
#plt.scatter(X_embedded1[:,0][mask_train],X_embedded1[:,1][mask_train],c=G.y[mask_train],cmap='tab10',vmax=9,alpha=0.5,marker='v',label='train') 
plt.legend()
plt.title('TSNE - LTFGW_GCN')
plt.show()


# %% CORNELL

df=pd.read_pickle('results/MLP_single_graph/cornell_seed20_lr0.05_n_temp15_n_nodes5_alpha0None_k1_localalphaFalse_drop0.8_shortpFalse_wd0.0005_hl64.pkl')
df_LTFGW=pd.read_pickle('see.pkl')
loss=df['loss']
validation=df['validation_accuracy']
train=df['train_accuracy']
loss_val=df['loss_validation']

loss_LTFGW=df_LTFGW['loss']
validation_LTFGW=df_LTFGW['validation_accuracy']
train_LTFGW=df_LTFGW['train_accuracy']
loss_val_LTFGW=df_LTFGW['loss_validation']

plt.figure(1)
plt.plot(loss,label='MLP')
plt.plot(loss_LTFGW,label='LTFGW-MLP')
plt.title('cornell - training loss')
plt.legend()

plt.figure(2)
plt.plot(loss_val,label='MLP')
plt.plot(loss_val_LTFGW,label='LTFGW-MLP')
plt.title('cornell - validation loss')
plt.legend()

plt.figure(3)
plt.plot(validation,label='MLP')
plt.plot(validation_LTFGW,label='LTFGW-MLP')
plt.title('cornell - validation accuracy')
plt.legend()

plt.figure(4)
plt.plot(train,label='MLP')
plt.plot(train_LTFGW,label='LTFGW-MLP')
plt.title('cornell - train accuracy')
plt.legend()



# %%
