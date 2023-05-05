from torch_geometric.loader import NeighborLoader, DataLoader
from architectures import GCN_LTFGW, GCN, MLP,LTFGW_GCN
from utils import get_dataset, get_filenames
from trainers import train, test, train_multi_graph
import os
import pandas as pd
import torch

torch.manual_seed(123456)

#%%Parameters to set

dataset_name='Toy_graph_multi'  #'Citeseer' or 'Toy_graph_single' or 'Toy_graph_multi'
model_name='LTFGW_GCN'  # 'GCN', 'GCN_LTFGW', 'LTFGW_GCN' or 'MLP'
save=True  #wether to save the parameters and the model
n_epoch=300 #number of epochs
training='multi_graph'     #'complete graph' or 'multi_graph' or 'mini_batch'
lr=0.01  #learning rate
weight_decay=5e-4

criterion = torch.nn.CrossEntropyLoss() 

num_seeds=1 #number of different seeds to train with

#%%Training and testing

Test_accuracy=0
seeds=torch.arange(20,20+num_seeds,1)

for seed in seeds:
    torch.manual_seed(seed)
    
    # load dataset
    dataset, n_classes, n_features=get_dataset(dataset_name)

    # init model
    if model_name=='LTFGW_GCN':
        model=LTFGW_GCN(n_classes=n_classes,n_features=n_features, n_templates=3,n_templates_nodes=3)

    elif model_name=='GCN_LTFGW':
        model=GCN_LTFGW(n_classes=n_classes,n_features=n_features, n_templates=6,n_templates_nodes=6, skip_connection=True)
  
    elif model_name=='MLP':
        model=MLP(n_classes=n_classes,n_features=n_features)

    elif model_name=='GCN':
        model=GCN(n_classes=n_classes,n_features=n_features)

    method=model_name+'_'+training
    filename_save, filename_best_model = get_filenames(dataset_name,method,seed)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

    best_val_perf=0

 #   if training=='mini_batch':
 #       train_loader = NeighborLoader(dataset,num_neighbors= [-1],
 #       batch_size=8,
 #       input_nodes=dataset.train_mask,shuffle=True)
 #       train_minibatch(model,train_loader,dataset,optimizer,criterion,n_epoch,save,filename_save,filename_best_model,best_val_perf)
        
    if training=='complete_graph':
        train(model,dataset,n_epoch,criterion, optimizer,save,filename_save,filename_best_model,best_val_perf)

    elif training=='multi_graph':
        generator = torch.Generator().manual_seed(seed.item())
        train_dataset,val_dataset,test_dataset=torch.utils.data.random_split(dataset,[600,200,200],generator=generator)
        train_loader=DataLoader(train_dataset,batch_size=100,shuffle=True)
        val_loader=DataLoader(val_dataset,batch_size=100,shuffle=True)
        train_multi_graph(model,criterion,optimizer,n_epoch,save,filename_save,filename_best_model,train_loader,val_loader)

