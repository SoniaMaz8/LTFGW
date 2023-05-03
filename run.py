from torch_geometric.loader import NeighborLoader
from architectures import GCN_LTFGW_parallel, GCN_2_layers, MLP,LTFGW_GCN
from data.convert_datasets import Citeseer_data
from trainers import train,train_minibatch, test
import os
import pandas as pd
import torch

torch.manual_seed(123456)

#%%Parameters to set

dataset_name='Toy_graph'  #'Citeseer' or 'Toy_graph'
model_name='LTFGW_GCN'  # 'GCN', 'GCN_LTFGW_parallel', 'LTFGW_GCN' or 'MLP'
save=True  #wether to save the parameters and the model
N_epoch=200 #number of epochs
training='complete_graph'     #'complete graph' or 'mini_batch' 
lr=0.01  #learning rate
weight_decay=5e-4

criterion = torch.nn.CrossEntropyLoss() 

num_seeds=0  #number of different seeds to train with

#%%Training and testing

Test_accuracy=0
seeds=torch.range(20,20+num_seeds,1)
for seed in seeds:
    torch.manual_seed(seed)

    if dataset_name=='Citeseer':
        dataset=Citeseer_data()
        n_classes=6
        filename_save='results/Citeseer'

    elif dataset_name=='Toy_graph':
        dataset=torch.load('data/toy_single_train.pt')
        n_classes=3
        filename_save='results/toy_graph'  

    if model_name=='GCN_LTFGW_parallel':
        model=GCN_LTFGW_parallel(n_classes=n_classes,N_features=dataset.num_features, N_templates=6,N_templates_nodes=6)
        filename_save=os.path.join( 'results','LTFGW',str(dataset_name)+ '.pkl')
        filename_best_model=os.path.join( 'results','LTFGW',str(dataset_name)+ '_best_valid.pt')

    elif model_name=='LTFGW_GCN':
        model=LTFGW_GCN(n_classes=n_classes,N_features=dataset.num_features)

    elif model_name=='MLP':
        model=MLP(n_classes=n_classes)

    elif model_name=='GCN':
        model=GCN_2_layers(n_classes=n_classes,N_features=dataset.num_features)
        filename_save=os.path.join( 'results','GCN',str(dataset_name)+ '.pkl')
        filename_best_model=os.path.join( 'results','GCN',str(dataset_name)+ '_best_valid.pt')       

    optimizer=torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    df=pd.read_pickle(filename_save)
    
    best_val_perf=df['max_val_accuracy'].max()

    if training=='mini_batch':
        train_loader = NeighborLoader(dataset,num_neighbors= [-1],
        batch_size=8,
        input_nodes=dataset.train_mask,shuffle=True)
        Loss, Train_acc,Val_acc=train_minibatch(model,train_loader,dataset,optimizer,criterion,N_epoch,save,filename_save,filename_best_model,best_val_perf,seed)
        
    elif training=='complete_graph':
        Loss, Train_acc, Val_acc=train(model,dataset,N_epoch,criterion, optimizer,save,filename_save,filename_best_model,best_val_perf,seed)
        
    test_acc=test(model,dataset)
    Test_accuracy+=test_acc

#print the average of the test accuracies over the seeds

print('mean_test_accuracy={}'.format( Test_accuracy/len(seeds)))        
    


