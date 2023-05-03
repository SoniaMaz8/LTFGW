from torch_geometric.loader import NeighborLoader
from architectures import GCN_LTFGW, GCN, MLP,LTFGW_GCN
from utils import get_dataset, get_filenames
from trainers import train,train_minibatch, test
import os
import pandas as pd
import torch

torch.manual_seed(123456)

#%%Parameters to set

dataset_name='Toy_graph'  #'Citeseer' or 'Toy_graph'
model_name='MLP'  # 'GCN', 'GCN_LTFGW', 'LTFGW_GCN' or 'MLP'
save=True  #wether to save the parameters and the model
N_epoch=200 #number of epochs
training='complete_graph'     #'complete graph' or 'mini_batch' 
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
    dataset, n_classes=get_dataset(dataset_name)

    # init model
    if model_name=='LTFGW_GCN':
        model=LTFGW_GCN(n_classes=n_classes,n_features=dataset.num_features, n_templates=6,n_templates_nodes=6)

    elif model_name=='GCN_LTFGW':
        model=GCN_LTFGW(n_classes=n_classes,n_features=dataset.num_features, n_templates=6,n_templates_nodes=6, skip_connection=True)
  
    elif model_name=='MLP':
        model=MLP(n_classes=n_classes,n_features=dataset.num_features)

    elif model_name=='GCN':
        model=GCN(n_classes=n_classes,n_features=dataset.num_features)

    method=model_name+'_'+training
    filename_save, filename_best_model = get_filenames(dataset_name,method,seed)

    print(filename_save)
    

    optimizer=torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

    best_val_perf=0
   # if save:
   #     df=pd.read_pickle(filename_save)
   #     best_val_perf=df['max_val_accuracy'].max()

    if training=='mini_batch':
        train_loader = NeighborLoader(dataset,num_neighbors= [-1],
        batch_size=8,
        input_nodes=dataset.train_mask,shuffle=True)
        Loss, Train_acc,Val_acc=train_minibatch(model,train_loader,dataset,optimizer,criterion,N_epoch,save,filename_save,filename_best_model,best_val_perf)
        
    elif training=='complete_graph':
        Loss, Train_acc, Val_acc=train(model,dataset,N_epoch,criterion, optimizer,save,filename_save,filename_best_model,best_val_perf)
        
    test_acc=test(model,dataset)
    Test_accuracy+=test_acc

#print the average of the test accuracies over the seeds

print('mean_test_accuracy={}'.format( Test_accuracy/len(seeds)))        
    


