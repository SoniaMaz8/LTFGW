#%%
import torch
from torch_geometric.loader import DataLoader
from main.trainers import *
from GNN.architectures import *
from main.utils import *
from torch_geometric.datasets import TUDataset

#%%

dataset = TUDataset(root='data/TUDataset', name='MUTAG')



#%%

n_graphs=len(dataset)
n_features=len(dataset[0].x[0])
n_classes=2

#%%

index=torch.randperm(n_graphs)
train_index=index[:int(0.6*n_graphs)]
val_index=index[int(0.6*n_graphs):int(0.8*n_graphs)]
test_index=index[int(0.8*n_graphs):]

dataset_train=[dataset[i] for i in train_index]
dataset_val=[dataset[i] for i in val_index]
dataset_test=[dataset[i] for i in test_index]

loader_train=DataLoader(dataset_train,batch_size=1)
loader_val=DataLoader(dataset_val,batch_size=1)
loader_test=DataLoader(dataset_test,batch_size=1)

lr=0.01
n_epoch=1000
save=True
n_templates=4
n_template_nodes=3
n_nodes=n_template_nodes
alpha0=None
k=1
dropout=0
wd=0.0005
hidden_layer=64
scheduler=False
log=False
shortest_path=False
seed=torch.tensor([20])
template_sizes=None
train_node_weights=True

mean_init=torch.mean(dataset[0].x)
std_init=torch.std(dataset[0].x)

model=TFGW_linear(0,hidden_layer,dropout,n_classes,n_features,n_templates,n_template_nodes,k,mean_init,std_init,alpha0,train_node_weights,shortest_path,template_sizes,log)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr)


filename_save, filename_best_model, filename_visus, filename_current_model, filename_templates, filename_alphas=get_filenames(
        'mutag',
        'TFGW_linear',
        lr,
        n_templates,
        n_nodes,
        alpha0,
        k,
        dropout,
        wd,
        hidden_layer,
        scheduler,
        seed,
        template_sizes=None,
        log=False)

train_multi_graph(model,
        criterion,
        optimizer,
        n_epoch,
        save,
        filename_save,
        filename_best_model,
        loader_train,
        loader_val,
        filename_visus)





# %%
