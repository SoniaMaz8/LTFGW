# %%

import pylab as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd
import numpy as np
from torch_geometric.utils import homophily


# %% Plot initial graph TSNE with connections between nodes

G = torch.load('data/anti_sbm.pt')

X_embedded = TSNE(n_components=2, perplexity=40).fit_transform(G.x)
edges = G.edge_index

plt.figure(figsize=(10, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
            c=G.y, cmap='tab10', vmax=9, alpha=0.5)

# plot connections
for i in range(len(edges[0])):
    plt.plot([X_embedded[edges[0, i], 0], X_embedded[edges[1, i], 0]], [
             X_embedded[edges[0, i], 1], X_embedded[edges[1, i], 1]], alpha=0.03, color='grey')

plt.title('TSNE antisbm')
plt.show()

# %% Plot latent embedding TSNE with shapes for train/validation

mutag = torch.load('data/mutag.pt')
generator = torch.Generator().manual_seed(20)

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


# %% anti SBM

df_GCN = pd.read_pickle(
    'results/GCN/anti_sbm1/20/performances/lr0.01_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.0_hl3_scheduler_False_logFalse.pkl')

df_LTFGW6 = pd.read_pickle(
    'MLP_LTFGW_linear.pkl')

df_LTFGW3 = pd.read_pickle(
    'results/MLP_LTFGW_linear/anti_sbm1/20/performances/lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.6_wd0.0005_hl3_scheduler_False_logFalse.pkl')


loss_GCN = df_GCN['loss']
validation_GCN = df_GCN['validation_accuracy']
train_GCN = df_GCN['train_accuracy']
loss_val_GCN = df_GCN['loss_validation']

loss_LTFGW6 = df_LTFGW6['loss']
validation_LTFGW6 = df_LTFGW6['validation_accuracy']
train_LTFGW6 = df_LTFGW6['train_accuracy']
loss_val_LTFGW6 =df_LTFGW6['loss_validation']

loss_LTFGW3 = df_LTFGW3['loss']
validation_LTFGW3 = df_LTFGW3['validation_accuracy']
train_LTFGW3 = df_LTFGW3['train_accuracy']
loss_val_LTFGW3 =df_LTFGW3['loss_validation']

plt.figure(1)
plt.plot(loss_LTFGW6, label='LTFGW_linear')
plt.plot(loss_GCN, label='GCN')
plt.plot(loss_LTFGW3, label='LTFGW 3')
plt.title('antisbm - training loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(2)
plt.plot(loss_val_LTFGW6, label='LTFGW_linear')
plt.plot(loss_val_GCN, label='GCN')
plt.plot(loss_val_LTFGW3, label='LTFGW 3')
plt.title('antisbm - validation loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(3)
plt.plot(validation_LTFGW6, label='LTFGW_linear')
plt.plot(validation_GCN, label='GCN')
plt.plot(validation_LTFGW3, label='LTFGW 3')
plt.title('antisbm - validation accuracy')
plt.xlabel('epochs')
plt.legend()

plt.figure(4)
plt.plot(train_LTFGW6, label='LTFGW_linear')
plt.plot(train_GCN, label='GCN')
plt.plot(train_LTFGW3, label='LTFGW 3')
plt.title('antisbm - train accuracy')
plt.xlabel('epochs')
plt.legend()

# %% CORNELL 2

df = pd.read_pickle(
    'results/MLP/cornell_directed/20/performances/lr0.05_n_temp1_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False.pkl')
df_LTFGW_12 = pd.read_pickle(
    'results/MLP_LTFGW/cornell_directed/20/performances/lr0.01_n_temp5_n_nodes3_alpha0None_k1_drop0.6_wd0.0005_hl3_scheduler_False_logFalse.pkl')

df_LTFGW_14 = pd.read_pickle(
    'results/MLP_LTFGW/cornell_directed/20/performances/lr0.01_n_temp5_n_nodes3_alpha0None_k1_drop0.6_wd0.0005_hl3_scheduler_False_logFalse.pkl')


df_LTFGW_16 = pd.read_pickle(
    'results/MLP_LTFGW/cornell_directed/20/performances/lr0.01_n_temp5_n_nodes3_alpha0None_k1_drop0.6_wd0.0005_hl3_scheduler_False_logFalse.pkl')


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
    'results/LTFGW/cornell_directed/20/performances/lr0.05_n_temp5_n_nodes3_alpha0None_k1_drop0.0_wd0.005_hl64_scheduler_False_logFalse.pkl')



df_LTFGW_34 = pd.read_pickle(
    'results/MLP_LTFGW/cornell_directed/20/performances/lr0.05_n_temp5_n_nodes3_alpha0None_k1_drop0.0_wd0.005_hl64_scheduler_False_logFalse.pkl')

df_LTFGW_44 = pd.read_pickle(
    'results/MLP_LTFGW/cornell_directed/20/performances/lr0.05_n_temp5_n_nodes3_alpha0None_k1_drop0.0_wd0.005_hl64_scheduler_False_logFalse.pkl')


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
plt.plot(loss, label='LTFGW-MLP with templates of different sizes')
#plt.plot(loss_LTFGW_24, label='LTFGW-MLP-relu - 5 templates, 4 nodes')
#plt.plot(loss_LTFGW_34, label='LTFGW-MLP-log - 5 templates, 4 nodes')
plt.plot(loss_LTFGW_44, label='LTFGW-MLP - 5 templates, 4 nodes')
plt.title('cornell - training loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(2)
plt.plot(loss_val, label='LTFGW-MLP with templates of different sizes')
#plt.plot(loss_val_LTFGW_24, label='LTFGW-MLP-relu - 5 templates, 4 nodes')
#plt.plot(loss_val_LTFGW_34, label='LTFGW-MLP-log - 5 templates, 4 nodes')
plt.plot(loss_val_LTFGW_44, label='LTFGW-MLP - 5 templates, 4 nodes')
plt.title('cornell - validation loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(3)
plt.plot(validation, label='LTFGW-MLP with templates of different sizes')
#plt.plot(validation_LTFGW_24, label='LTFGW-MLP-relu - 5 templates, 4 nodes')
#plt.plot(validation_LTFGW_34, label='LTFGW-MLP-log - 5 templates, 4 nodes')
plt.plot(validation_LTFGW_44, label='LTFGW-MLP - 5 templates, 4 nodes')
plt.title('cornell - validation accuracy')
plt.xlabel('epochs')
plt.legend()

plt.figure(4)
plt.plot(train, label='LTFGW-MLP with templates of different sizes')
#plt.plot(train_LTFGW_24,label='LTFGW-MLP-relu - 5 templates, 4 nodes')
#plt.plot(train_LTFGW_34, label='LTFGW-MLP-log - 5 templates, 4 nodes')
plt.plot(train_LTFGW_44, label='LTFGW-MLP - 5 template, 4 nodes')
plt.title('cornell - train accuracy')
plt.xlabel('epochs')
plt.legend()

# %% Plot template and alpha evolution

df = torch.load(
    'results_3:07/LTFGW_MLP_dropout/cornell_directed/20/templates/lr0.05_n_temp2_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False_logFalse.pkl')
norms = []
plt.figure(1)
for i in range(len(df) - 1):
    norm = torch.abs((torch.linalg.matrix_norm(
        df[i][0]) - torch.linalg.matrix_norm(df[i + 1][0]))) / torch.linalg.matrix_norm(df[i][0])
    norms.append(norm.item())

plt.plot(norms)
plt.title('norm of the difference between two consecutive templates, 2 templates, 6 nodes')
plt.xlabel('epochs')

plt.figure(2)
df = torch.load(
    'results_3:07/LTFGW_MLP_dropout/cornell_directed/20/alphas/lr0.05_n_temp2_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_scheduler_False_logFalse.pkl')

print(df)

alphas = []
for i in range(len(df)):
    alpha = torch.sigmoid(df[i]).item()
    alphas.append(alpha)

plt.plot(alphas)
plt.title('evolution of alpha (trade off parameter for FGW in LTFGW-MLP), 2 templates, 6 nodes')
plt.xlabel('epochs')

# %%

df = torch.load(
    'results/LTFGW_MLP_dropout/cornell/20/templates/n_temp0.0005_n_nodes1_alpha0183_kNone_drop1_wd0.8_hl0.0005.pkl')
print(df)
# %% Box plots

data12=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp2_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data14=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp4_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data16=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp6_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data18=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp8_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data18=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp10_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')

plt.boxplot([data12,data14,data16],labels=['1-2','1-4','1-6'])


# %%

MLP=np.loadtxt('results/MLP/cornell_directed/test_seed20_lr0.05_n_temp1_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data26=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp2_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data46=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp4_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data66=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp6_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data86=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp8_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data106=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp10_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
#data_log=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp5_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logTrue.csv')


bp=plt.boxplot([MLP,data26,data46,data66,data86,data106],labels=['MLP','2-6','4-6','6-6','8-6','10-6'],patch_artist=True,showmeans=True)
bp['boxes'][0].set_facecolor('red')
plt.title('boxplots of the accuracy on test for 10 different train/test splits, for different numbers of templates')
plt.xlabel('number of templates - number of nodes in each template')
# %%

MLP=np.loadtxt('results/MLP/cornell_directed/test_seed20_lr0.05_n_temp1_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')

data24=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed20_lr0.05_n_temp2_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data23=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed29_lr0.05_n_temp2_n_nodes3_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data25=np.loadtxt('results/LTFGW_MLP_dropout/cornell_directed/test_seed29_lr0.05_n_temp2_n_nodes5_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data26=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp2_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
bp=plt.boxplot([MLP,data23,data24,data25,data26],labels=['MLP','2-3','2-4','2-5','2-6'],patch_artist=True,showmeans=True)
plt.xlabel('number of templates - number of nodes in each template')
plt.title('boxplots of the accuracy on test for 10 different train/test splits, for different numbers of templates nodes')

# %%

MLP=np.loadtxt('results/MLP/cornell_directed/test_seed20_lr0.05_n_temp1_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data_log=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed21_lr0.05_n_temp5_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logTrue.csv')
bp=plt.boxplot([MLP,data_log],labels=['MLP','log'],patch_artist=True,showmeans=True)
plt.title('boxplots of the accuracy on test for 10 different train/test splits, for different numbers of templates')

# %%

MLP=np.loadtxt('results/MLP/cornell_directed/test_seed20_lr0.05_n_temp1_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64.csv')
data26=np.loadtxt('results_3:07/LTFGW_MLP_dropout/cornell_directed/test_seed29_lr0.05_n_temp2_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data46=np.loadtxt('results_3:07/LTFGW_MLP_dropout/cornell_directed/test_seed29_lr0.05_n_temp4_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data66=np.loadtxt('results_3:07/LTFGW_MLP_dropout/cornell_directed/test_seed29_lr0.05_n_temp6_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data86=np.loadtxt('results_3:07/LTFGW_MLP_dropout/cornell_directed/test_seed29_lr0.05_n_temp8_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
data106=np.loadtxt('results_3:07/LTFGW_MLP_dropout/cornell_directed/test_seed29_lr0.05_n_temp10_n_nodes6_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logFalse.csv')
#data_log=np.loadtxt('results_3:07/LTFGW_MLP_dropout/cornell_directed/test_seed29_lr0.05_n_temp5_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logTrue.csv')


bp=plt.boxplot([MLP,data26,data46,data66,data86,data106,data_log],labels=['MLP','2-6','4-6','6-6','8-6','10-6','5-4-log'],patch_artist=True,showmeans=True)
bp['boxes'][0].set_facecolor('red')
plt.title('boxplots of the accuracy on test for 10 different train/test splits, for different numbers of templates')
plt.xlabel('number of templates - number of nodes in each template')

# %% semi_relaxed

df = pd.read_pickle(
    'results/GCN/anti_sbm1/21/performances/lr0.01_n_temp1_n_nodes2_alpha0None_k1_drop0.0_wd0.005_hl3_scheduler_False_logFalse.pkl')
df_LTFGW_12 = pd.read_pickle(
    'results/MLP_LTFGW/anti_sbm1/21/performances/lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_scheduler_False_logFalse.pkl')


loss = df['loss']
validation = df['validation_accuracy']
train = df['train_accuracy']
loss_val = df['loss_validation']

loss_LTFGW_12 = df_LTFGW_12['loss']
validation_LTFGW_12 = df_LTFGW_12['validation_accuracy']
train_LTFGW_12 = df_LTFGW_12['train_accuracy']
loss_val_LTFGW_12 = df_LTFGW_12['loss_validation']


plt.figure(1)
plt.plot(loss, label='GCN')
plt.plot(loss_LTFGW_12, label='LTFGW-linear - 3 templates, 3 nodes')
plt.title('anti sbm - training loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(2)
plt.plot(loss_val, label='GCN')
plt.plot(loss_val_LTFGW_12, label='LTFGW-linear - 3 templates, 3 nodes')
plt.title('anti sbm - validation loss')
plt.xlabel('epochs')
plt.legend()

plt.figure(3)
plt.plot(validation, label='GCN')
plt.plot(validation_LTFGW_12, label='LTFGW-linear - 3 templates, 3 nodes')
plt.title('anti sbm - validation accuracy')
plt.xlabel('epochs')
plt.legend()

plt.figure(4)
plt.plot(train, label='GCN')
plt.plot(train_LTFGW_12,label='LTFGW-linear - 3 template, 3 nodes')
plt.title('anti sbm - train accuracy')
plt.xlabel('epochs')
plt.legend()
# %% anti_sbm
#MLP=np.loadtxt('results/MLP/anti_sbm/test_seed29_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.05_hl3_schedulerFalse_logFalse.csv')
#GCN=np.loadtxt('results/GCN/anti_sbm/test_seed29_lr0.01_n_temp3_n_nodes3_alpha0None_k1_drop0.2_wd0.005_hl3_schedulerFalse_logFalse.csv')

data1=np.loadtxt('results/GCN/anti_sbm1/test_seed29_lr0.01_n_temp1_n_nodes2_alpha0None_k1_drop0.0_wd0.005_hl3_schedulerFalse_logFalse.csv')
data2=np.loadtxt('results/GCN/anti_sbm2/test_seed29_lr0.01_n_temp1_n_nodes2_alpha0None_k1_drop0.0_wd0.005_hl3_schedulerFalse_logFalse.csv')
data3=np.loadtxt('results/GCN/anti_sbm3/test_seed29_lr0.01_n_temp1_n_nodes2_alpha0None_k1_drop0.0_wd0.005_hl3_schedulerFalse_logFalse.csv')
data4=np.loadtxt('results/GCN/anti_sbm4/test_seed29_lr0.01_n_temp1_n_nodes2_alpha0None_k1_drop0.0_wd0.005_hl3_schedulerFalse_logFalse.csv')
data5=np.loadtxt('results/GCN/anti_sbm5/test_seed29_lr0.01_n_temp1_n_nodes2_alpha0None_k1_drop0.0_wd0.005_hl3_schedulerFalse_logFalse.csv')
data6=np.loadtxt('results/GCN/anti_sbm6/test_seed29_lr0.01_n_temp1_n_nodes2_alpha0None_k1_drop0.0_wd0.005_hl3_schedulerFalse_logFalse.csv')
#data0_005_0_0005=np.loadtxt('results/LTFGW/anti_sbm/test_seed29_lr0.005_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.0005_hl3_schedulerFalse_logFalse.csv')
#data0_05_0_00005=np.loadtxt('results/LTFGW/anti_sbm/test_seed29_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_schedulerFalse_logFalse.csv')
#data0_005_0_00005=np.loadtxt('results/LTFGW/anti_sbm/test_seed29_lr0.005_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_schedulerFalse_logFalse.csv')
#data64=np.loadtxt('results/LTFGW_MLP_dropout/anti_sbm/test_seed25_lr0.05_n_temp6_n_nodes4_alpha0None_k1_drop0.0_wd0.05_hl3_schedulerFalse_logFalse.csv')
#data_log=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp5_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logTrue.csv')


bp=plt.boxplot([data1,data2,data3,data4,data5,data6],labels=['1','2','3','4','5','6'],patch_artist=True,showmeans=True)
plt.xlabel('increasing heterophily')
plt.title('GCN - antisbm - boxplots of the accuracy on test for 10 different train/test splits')
# %%

#MLP=np.loadtxt('results/MLP/anti_sbm/test_seed29_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.05_hl3_schedulerFalse_logFalse.csv')
#GCN=np.loadtxt('results/GCN/anti_sbm/test_seed29_lr0.01_n_temp3_n_nodes3_alpha0None_k1_drop0.2_wd0.005_hl3_schedulerFalse_logFalse.csv')

data1=np.loadtxt('results/MLP_LTFGW/anti_sbm1/test_seed29_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_schedulerFalse_logFalse.csv')
data2=np.loadtxt('results/MLP_LTFGW/anti_sbm2/test_seed29_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_schedulerFalse_logFalse.csv')
data3=np.loadtxt('results/MLP_LTFGW/anti_sbm3/test_seed29_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_schedulerFalse_logFalse.csv')
data4=np.loadtxt('results/MLP_LTFGW/anti_sbm4/test_seed29_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_schedulerFalse_logFalse.csv')
data5=np.loadtxt('results/MLP_LTFGW/anti_sbm5/test_seed29_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_schedulerFalse_logFalse.csv')
data6=np.loadtxt('results/MLP_LTFGW/anti_sbm6/test_seed29_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_schedulerFalse_logFalse.csv')
#data0_005_0_0005=np.loadtxt('results/LTFGW/anti_sbm/test_seed29_lr0.005_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.0005_hl3_schedulerFalse_logFalse.csv')
#data0_05_0_00005=np.loadtxt('results/LTFGW/anti_sbm/test_seed29_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_schedulerFalse_logFalse.csv')
#data0_005_0_00005=np.loadtxt('results/LTFGW/anti_sbm/test_seed29_lr0.005_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd5e-05_hl3_schedulerFalse_logFalse.csv')
#data64=np.loadtxt('results/LTFGW_MLP_dropout/anti_sbm/test_seed25_lr0.05_n_temp6_n_nodes4_alpha0None_k1_drop0.0_wd0.05_hl3_schedulerFalse_logFalse.csv')
#data_log=np.loadtxt('results_2:07/LTFGW_MLP_dropout_relu/cornell_directed/test_seed29_lr0.05_n_temp5_n_nodes4_alpha0None_k1_drop0.8_wd0.0005_hl64_schedulerFalse_logTrue.csv')


bp=plt.boxplot([data1,data2,data3,data4,data5,data6],labels=['1','2','3','4','5','6'],patch_artist=True,showmeans=True)
plt.xlabel('increasing heterophily')
plt.title('LTFGW-Linear - antisbm - boxplots of the accuracy on test for 10 different train/test splits')
# %%

data1=np.loadtxt('results/MLP_LTFGW/anti_sbm1/test_seed27_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.0005_hl3_schedulerFalse_logFalse.csv')
data2=np.loadtxt('results/MLP_LTFGW/anti_sbm2/test_seed27_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.0005_hl3_schedulerFalse_logFalse.csv')
data3=np.loadtxt('results/MLP_LTFGW/anti_sbm3/test_seed27_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.0005_hl3_schedulerFalse_logFalse.csv')
data4=np.loadtxt('results/MLP_LTFGW/anti_sbm4/test_seed27_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.0005_hl3_schedulerFalse_logFalse.csv')
data5=np.loadtxt('results/MLP_LTFGW/anti_sbm5/test_seed27_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.0005_hl3_schedulerFalse_logFalse.csv')
data6=np.loadtxt('results/MLP_LTFGW/anti_sbm6/test_seed27_lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.0_wd0.0005_hl3_schedulerFalse_logFalse.csv')

data1_GCN=np.loadtxt('results/GCN/anti_sbm1/test_seed29_lr0.01_n_temp3_n_nodes3_alpha0None_k1_drop0.6_wd0.0_hl3_schedulerFalse_logFalse.csv')
data2_GCN=np.loadtxt('results/GCN/anti_sbm2/test_seed29_lr0.01_n_temp3_n_nodes3_alpha0None_k1_drop0.6_wd0.0_hl3_schedulerFalse_logFalse.csv')
data3_GCN=np.loadtxt('results/GCN/anti_sbm3/test_seed29_lr0.01_n_temp3_n_nodes3_alpha0None_k1_drop0.6_wd0.0_hl3_schedulerFalse_logFalse.csv')
data4_GCN=np.loadtxt('results/GCN/anti_sbm4/test_seed29_lr0.01_n_temp3_n_nodes3_alpha0None_k1_drop0.6_wd0.0_hl3_schedulerFalse_logFalse.csv')
data5_GCN=np.loadtxt('results/GCN/anti_sbm5/test_seed29_lr0.01_n_temp3_n_nodes3_alpha0None_k1_drop0.6_wd0.0_hl3_schedulerFalse_logFalse.csv')
data6_GCN=np.loadtxt('results/GCN/anti_sbm6/test_seed29_lr0.01_n_temp3_n_nodes3_alpha0None_k1_drop0.6_wd0.0_hl3_schedulerFalse_logFalse.csv')

graph1=torch.load('data/anti_sbm1.pt')
graph2=torch.load('data/anti_sbm2.pt')
graph3=torch.load('data/anti_sbm3.pt')
graph4=torch.load('data/anti_sbm4.pt')
graph5=torch.load('data/anti_sbm5.pt')
graph6=torch.load('data/anti_sbm6.pt')

hom1=homophily(graph1.edge_index,graph1.y)
hom2=homophily(graph2.edge_index,graph2.y)
hom3=homophily(graph3.edge_index,graph3.y)
hom4=homophily(graph4.edge_index,graph4.y)
hom5=homophily(graph5.edge_index,graph5.y)
hom6=homophily(graph6.edge_index,graph6.y)

LTFGW=[data1,data2,data3,data4,data5,data6]
GCN=[data1_GCN,data2_GCN,data3_GCN,data4_GCN,data5_GCN,data6_GCN]

ticks=["{:.3f}".format(hom1),"{:.3f}".format(hom2),"{:.3f}".format(hom3),"{:.3f}".format(hom4),"{:.3f}".format(hom5),"{:.3f}".format(hom6)]

bp=plt.boxplot(LTFGW,patch_artist=True,showmeans=True,positions=np.array(
    np.arange(len(LTFGW)))*3.0+0.35,widths=0.6)

bp_GCN=plt.boxplot(GCN,patch_artist=True,showmeans=True,positions=np.array(
    np.arange(len(GCN)))*3.0-0.35,widths=0.6)

def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()  

define_box_properties(bp, '#D7191C', 'LTFGW')
define_box_properties(bp_GCN, '#2C7BB6', 'GCN')

plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks)

plt.xlabel('increasing homophily')
plt.title('LTFGW - antisbm - boxplots of the accuracy on test for 10 different train/test splits')
# %% Plot templates

parameters = torch.load('results/MLP_LTFGW/anti_sbm1/20/best_model/lr0.05_n_temp3_n_nodes3_alpha0None_k1_drop0.6_wd0.0005_hl3_scheduler_False_logFalse.pkl')
parameters[checkpoint['model_state_dict']]


