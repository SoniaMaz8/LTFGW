from torch_geometric.loader import NeighborLoader
from architectures import GCN_LTFGW, GCN_3_layers
import torch
from data.convert_datasets import Citeseer_data
from trainers import train,train_minibatch, test
import os
import pandas as pd

torch.manual_seed(123456)

#%%Parameters to set

dataset_name='citeseer'  #'citeseer' or 'toy_graph'
model_name='LTFGW'  #'LTFGW' or 'GCN'
save_parameters=True  #wether to save the parameters and the model
N_epoch=200  #number of epochs
training='complete_graph'     #'complete graph' or 'mini_batch' 
lr=0.01  #learning rate
weight_decay=5e-4

criterion = torch.nn.CrossEntropyLoss() 

num_seeds=0  #number of different seeds to train with

#%%Training and testing

Test_accuracy=0
seeds=torch.range(40,40+num_seeds,1)
for seed in seeds:
    torch.manual_seed(seed)

    if dataset_name=='citeseer':
        dataset=Citeseer_data()
        n_classes=6
        filename_save='results/Citeseer'

    elif dataset_name=='toy_graph':
        dataset=torch.load('data/toy_graph1.pt')
        n_classes=3
        filename_save='results/toy_graph'  

    if model_name=='LTFGW':
        model=GCN_LTFGW(n_classes=n_classes,N_features=dataset.num_features, N_templates=6,N_templates_nodes=6)
        filename_save=os.path.join( filename_save,'LTFGW.csv')

    elif model_name=='GCN':
        model=GCN_3_layers(n_classes=n_classes,N_features=dataset.num_features,dropout=0.6)
        filename_save=os.path.join(filename_save,'GCN.csv')
    
    optimizer=torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

    if training=='mini_batch':
        train_loader = NeighborLoader(dataset,num_neighbors= [-1],
        batch_size=8,
        input_nodes=dataset.train_mask,shuffle=True)
        Loss, Train_acc=train_minibatch(model,train_loader,dataset,optimizer,criterion,N_epoch)
        
    elif training=='complete_graph':
        Loss, Train_acc, Val_acc=train(model,dataset,N_epoch,criterion, optimizer)
        
    test_acc=test(model,dataset)
    Test_accuracy+=test_acc

    if save_parameters:
        df = pd.read_csv(filename_save)
        new_row={'seed':seed.item(), 'loss': Loss,'train_accuracy':Train_acc ,'validation_accuracy': Val_acc,'test_accuracy':test_acc,'model_parameters': model.state_dict()}
        df.loc[len(df)]=new_row
        df.to_csv(filename_save)


#print the average of the test accuracies over the seeds

print('mean_test_accuracy={}'.format( Test_accuracy/len(seeds)))        
    











