# %%

import pylab as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd
from GNN.utils import moving_average
import numpy as np


# %% Plot initial graph TSNE with connections between nodes

G = torch.load('data/mutag.pt')
G = G[0]
print(G.x)

X_embedded = TSNE(n_components=7, perplexity=5).fit_transform(G.x)
edges = G.edge_index

plt.figure(figsize=(10, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
            c=G.y, cmap='tab10', vmax=9, alpha=0.5)

# plot connections
for i in range(len(edges[0])):
    plt.plot([X_embedded[edges[0, i], 0], X_embedded[edges[1, i], 0]], [
             X_embedded[edges[0, i], 1], X_embedded[edges[1, i], 1]], alpha=0.01, color='grey')

plt.title('TSNE initial')
plt.show()

# %% Plot latent embedding TSNE with shapes for train/validation

mutag = torch.load('data/mutag.pt')
generator = torch.Generator().manual_seed(20)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    mutag, [112, 38, 38], generator=generator)
G = torch.load('g.pt')

plt.figure(2, figsize=(10, 8))

latent = pd.read_pickle('results/LTFGW_GCN_multi_graph/mutag_seed20_visus.pkl')
latent = torch.tensor(latent.values)
latent = latent[:, 1:]


X_embedded1 = TSNE(n_components=2, perplexity=30).fit_transform(latent)

edges = G.edge_index

# masks for train and test

# mask_val=G.val_mask.numpy()
# mask_train=(1-mask_val)==1

print(len(G.y))

plt.scatter(X_embedded1[:, 0], X_embedded1[:, 1], c=G.y,
            cmap='tab10', vmax=9, alpha=0.5, marker='o')
# plt.scatter(X_embedded1[:,0][mask_train],X_embedded1[:,1][mask_train],c=G.y[mask_train],cmap='tab10',vmax=9,alpha=0.5,marker='v',label='train')
plt.legend()
plt.title('TSNE - LTFGW_GCN')
plt.show()


# %% CORNELL

df = pd.read_pickle(
    'results/MLP/cornell/21/performances/lr0.0005_n_temp1_n_nodes2_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')
df_LTFGW_undirected = pd.read_pickle(
    'results/LTFGW_MLP_dropout/cornell/21/performances/lr0.0005_n_temp1_n_nodes2_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')

df_LTFGW_directed = pd.read_pickle(
    'results/LTFGW_MLP_dropout/cornell_directed/21/performances/lr0.0005_n_temp1_n_nodes2_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')


loss = df['loss']
validation = df['validation_accuracy']
train = df['train_accuracy']
loss_val = df['loss_validation']

loss_LTFGW_undirected = df_LTFGW_undirected['loss']
validation_LTFGW_undirected = df_LTFGW_undirected['validation_accuracy']
train_LTFGW_undirected = df_LTFGW_undirected['train_accuracy']
loss_val_LTFGW_undirected = df_LTFGW_undirected['loss_validation']

loss_LTFGW_directed = df_LTFGW_directed['loss']
validation_LTFGW_directed = df_LTFGW_directed['validation_accuracy']
train_LTFGW_directed = df_LTFGW_directed['train_accuracy']
loss_val_LTFGW_directed = df_LTFGW_directed['loss_validation']


plt.figure(1)
plt.plot(loss, label='MLP')
plt.plot(loss_LTFGW_undirected, label='LTFGW-MLP-undirected')
plt.plot(loss_LTFGW_directed, label='LTFGW-MLP-directed')
plt.title('cornell - training loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(2)
plt.plot(loss_val, label='MLP')
plt.plot(loss_val_LTFGW_undirected, label='LTFGW-MLP-undirected')
plt.plot(loss_val_LTFGW_directed, label='LTFGW-MLP-directed')
plt.title('cornell - validation loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(3)
plt.plot(validation, label='MLP')
plt.plot(validation_LTFGW_undirected, label='LTFGW-MLP-undirected')
plt.plot(validation_LTFGW_directed, label='LTFGW-MLP-directed')
plt.title('cornell - validation accuracy')
plt.xlabel('epochs')
plt.legend()

plt.figure(4)
plt.plot(train, label='MLP')
plt.plot(train_LTFGW_undirected, label='LTFGW-MLP-undirected')
plt.plot(train_LTFGW_directed, label='LTFGW-MLP-directed')
plt.title('cornell - train accuracy')
plt.xlabel('epochs')
plt.legend()

# %% CORNELL 2

df = pd.read_pickle(
    'results/MLP/cornell/22/performances/lr0.05_n_temp1_n_nodes2_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')
df_LTFGW_12 = pd.read_pickle(
    'results/LTFGW_MLP_dropout/cornell_directed/20/performances/lr0.05_n_temp1_n_nodes2_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')

df_LTFGW_14 = pd.read_pickle(
    'results/LTFGW_MLP_dropout/cornell_directed/20/performances/lr0.05_n_temp1_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')

df_LTFGW_16 = pd.read_pickle(
    'results/LTFGW_MLP_dropout/cornell_directed/20/performances/lr0.05_n_temp1_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')


loss = df['loss']
validation = df['validation_accuracy']
train = df['train_accuracy']
loss_val = df['loss_validation']

loss_LTFGW_12 = df_LTFGW_12['loss']
validation_LTFGW_12 = df_LTFGW_12['validation_accuracy']
train_LTFGW_12 = df_LTFGW_12['train_accuracy']
loss_val_LTFGW_12 = df_LTFGW_12['loss_validation']


loss_LTFGW_14 = df_LTFGW_14['loss']
validation_LTFGW_14 = df_LTFGW_14['validation_accuracy']
train_LTFGW_14 = df_LTFGW_14['train_accuracy']
loss_val_LTFGW_14 = df_LTFGW_14['loss_validation']

loss_LTFGW_16 = df_LTFGW_16['loss']
validation_LTFGW_16 = df_LTFGW_16['validation_accuracy']
train_LTFGW_16 = df_LTFGW_16['train_accuracy']
loss_val_LTFGW_16 = df_LTFGW_16['loss_validation']




plt.figure(1)
plt.plot(loss, label='MLP')
plt.plot(loss_LTFGW_12, label='LTFGW-MLP - 1 template, 2 nodes')
plt.plot(loss_LTFGW_14, label='LTFGW-MLP - 1 template, 4 nodes')
plt.plot(loss_LTFGW_16, label='LTFGW-MLP - 1 template, 6 nodes')
plt.title('cornell - training loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(2)
plt.plot(loss_val, label='MLP')
plt.plot(loss_val_LTFGW_12, label='LTFGW-MLP - 1 template, 2 nodes')
plt.plot(loss_val_LTFGW_14, label='LTFGW-MLP - 1 template, 4 nodes')
plt.plot(loss_val_LTFGW_16, label='LTFGW-MLP - 1 template, 6 nodes')
plt.title('cornell - validation loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(3)
plt.plot(validation, label='MLP')
plt.plot(validation_LTFGW_12, label='LTFGW-MLP - 1 template, 2 nodes')
plt.plot(validation_LTFGW_14, label='LTFGW-MLP - 1 template, 4 nodes')
plt.plot(validation_LTFGW_16, label='LTFGW-MLP - 1 template, 6 nodes')
plt.title('cornell - validation accuracy')
plt.xlabel('epochs')
plt.legend()

plt.figure(4)
plt.plot(train, label='MLP')
plt.plot(train_LTFGW_12,label='LTFGW-MLP - 1 template, 2 nodes')
plt.plot(train_LTFGW_14, label='LTFGW-MLP - 1 template, 4 nodes')
plt.plot(train_LTFGW_16, label='LTFGW-MLP - 1 template, 6 nodes')
plt.title('cornell - train accuracy')
plt.xlabel('epochs')
plt.legend()

# %% CORNELL 3

df = pd.read_pickle(
    'results/MLP/cornell_directed/20/performances/lr0.05_n_temp1_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')
df_LTFGW_24 = pd.read_pickle(
    'results/LTFGW_MLP_dropout/cornell_directed/20/performances/lr0.05_n_temp10_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')

df_LTFGW_34 = pd.read_pickle(
    'see.pkl')

df_LTFGW_44 = pd.read_pickle(
    'results/LTFGW_MLP_dropout/cornell_directed/23/performances/lr0.05_n_temp10_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')


loss = df['loss']
validation = df['validation_accuracy']
train = df['train_accuracy']
loss_val = df['loss_validation']

loss_LTFGW_24 = df_LTFGW_24['loss']
validation_LTFGW_24 = df_LTFGW_24['validation_accuracy']
train_LTFGW_24 = df_LTFGW_24['train_accuracy']
loss_val_LTFGW_24 = df_LTFGW_24['loss_validation']


loss_LTFGW_34 = df_LTFGW_34['loss']
validation_LTFGW_34 = df_LTFGW_34['validation_accuracy']
train_LTFGW_34 = df_LTFGW_34['train_accuracy']
loss_val_LTFGW_34 = df_LTFGW_34['loss_validation']

loss_LTFGW_44 = df_LTFGW_44['loss']
validation_LTFGW_44 = df_LTFGW_44['validation_accuracy']
train_LTFGW_44 = df_LTFGW_44['train_accuracy']
loss_val_LTFGW_44 = df_LTFGW_44['loss_validation']




plt.figure(1)
plt.plot(loss, label='MLP')
plt.plot(loss_LTFGW_24, label='LTFGW-MLP - 10 templates, 4 nodes')
plt.plot(loss_LTFGW_34, label='LTFGW-MLP - 3 templates, 4 nodes')
#plt.plot(loss_LTFGW_44, label='LTFGW-MLP - 4 templates, 6 nodes')
plt.title('cornell - training loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(2)
plt.plot(loss_val, label='MLP')
plt.plot(loss_val_LTFGW_24, label='LTFGW-MLP - 10 templates, 4 nodes')
plt.plot(loss_val_LTFGW_34, label='LTFGW-MLP - 3 templates, 4 nodes')
#plt.plot(loss_val_LTFGW_44, label='LTFGW-MLP - 4 templates, 4 nodes')
plt.title('cornell - validation loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(3)
plt.plot(validation, label='MLP')
plt.plot(validation_LTFGW_24, label='LTFGW-MLP - 10 templates, 4 nodes')
plt.plot(validation_LTFGW_34, label='LTFGW-MLP - 3 templates, 4 nodes')
#plt.plot(validation_LTFGW_44, label='LTFGW-MLP - 4 templates, 4 nodes')
plt.title('cornell - validation accuracy')
plt.xlabel('epochs')
plt.legend()

plt.figure(4)
plt.plot(train, label='MLP')
plt.plot(train_LTFGW_24,label='LTFGW-MLP - 10 templates, 4 nodes')
plt.plot(train_LTFGW_34, label='LTFGW-MLP - 3 template, 4 nodes')
#plt.plot(train_LTFGW_44, label='LTFGW-MLP - 4 template, 4 nodes')
plt.title('cornell - train accuracy')
plt.xlabel('epochs')
plt.legend()

# %% Plot template and alpha evolution

df = torch.load(
    'results/MLP/cornell/20/performances/lr0.0005_n_temp15_n_nodes5_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')
norms = []
plt.figure(1)
for i in range(len(df) - 1):
    norm = torch.abs((torch.linalg.matrix_norm(
        df[i][0]) - torch.linalg.matrix_norm(df[i + 1][0]))) / torch.linalg.matrix_norm(df[i][0])
    norms.append(norm.item())

plt.plot(norms)
plt.title('norm of the difference between two consecutive templates')
plt.xlabel('epochs')

plt.figure(2)
df = torch.load(
    'results/LTFGW_MLP_dropout/cornell/20/alphas/lr0.0005_n_temp1_n_nodes2_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')

print(df)

alphas = []
for i in range(len(df)):
    alpha = torch.sigmoid(df[i]).item()
    alphas.append(alpha)

plt.plot(alphas)
plt.title('evolution of alpha (trade off parameter for FGW in LTFGW-MLP)')
plt.xlabel('epochs')

# %%

df = torch.load(
    'results/LTFGW_MLP_dropout/cornell/20/templates/n_temp0.0005_n_nodes1_alpha0183_kNone_drop1_wd0.8_hl0.0005.pkl')
print(df)
# %% Box plots

data12=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp1_n_nodes2_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data14=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp1_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data16=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp1_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
plt.boxplot([data12,data14,data16],labels=['1-2','1-4','1-6'])


# %%

MLP=np.loadtxt('results/MLP/cornell_directed/test_seed20_lr0.05_n_temp1_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')[0:7]
data24=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp2_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data34=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp3_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data44=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp4_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data54=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp5_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data64=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp6_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data74=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp7_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data84=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp8_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data94=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp9_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data104=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp10_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')


bp=plt.boxplot([MLP,data24,data34,data44,data54,data64,data74,data84,data94,data104],labels=['MLP','2-4','3-4','4-4','5-4','6-4','7-4','8-4','9-4','10-4'],patch_artist=True,showmeans=True)
bp['boxes'][0].set_facecolor('red')
plt.title('boxplots of the accuracy on test for 7 different train/test splits, for different numbers of templates')
plt.xlabel('number of templates - number of nodes in each template')
# %%
