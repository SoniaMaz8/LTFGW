#%%

import pylab as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd



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

latent=pd.read_csv('results/LTFGW_GCN_multi_graph/mutag_seed20_visus.pkl')
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



#%% Plot train and validation

df=pd.read_pickle('results/LTFGW_GCN_multi_graph/mutag_seed20.pkl')
loss=df['loss']
validation=df['validation_accuracy']
train=df['train_accuracy']
df_GCN=pd.read_pickle('results/GCN_multi_graph/mutag_seed20.pkl')
loss_GCN=df_GCN['loss']
validation_GCN=df_GCN['validation_accuracy']
train_GCN=df_GCN['train_accuracy']

plt.figure(1)
plt.plot(loss,label='loss LTFGW')
plt.plot(loss_GCN,label='loss GCN')
plt.title('mutag - 60/20/20')
plt.legend()
plt.savefig('loss.png')
plt.show()


plt.figure(2)
plt.plot(validation,label='validation LTFGW')
plt.plot(validation_GCN,label='validation GCN')
plt.title('mutag - 60/20/20')
plt.legend()
plt.savefig('validation.png')
plt.show()


plt.figure(3)
plt.plot(train,label='train LTFGW')
plt.plot(train_GCN,label='train GCN')
plt.title('mutag - 60/20/20')
plt.legend()
plt.savefig('train.png')
plt.show()


#%%
