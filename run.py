#%%


from torch_geometric.loader import NeighborLoader, DataLoader
from GNN.architectures import *

from GNN.utils import get_dataset, get_filenames
from main.trainers import train, test, train_multi_graph, test_multigraph
import os
import torch
import numpy as np
import torch_geometric.transforms as T

#%%Parameters to set

dataset_name='cornell'  #'Citeseer' or 'Toy_graph_single' or 'Toy_graph_multi' or 'mutag' or 'cornell'
model_name='LTFGW_MLP'  # 'GCN', 'GCN_LTFGW', 'LTFGW_GCN' or 'MLP'  or 'LTFGW_MLP'
save=True  #wether to save the parameters and the model
n_epoch=1000 #number of epochs
training='complete_graph'     #'complete_graph' or 'multi_graph' or 'mini_batch'
lr=0.05 #learning rate
weight_decay=5e-4
random_split=[112,38,38]
#random_split=[600,200,200]

n_templates=10
n_templates_nodes=3

hidden_layer=64
n_hidden_layer=0

batch_size=5
first_seed=21

train_node_weights=True

criterion = torch.nn.CrossEntropyLoss() 

num_seeds=10 #number of different seeds to train with

#%%Training and testing

Test_accuracy=[]
#SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]
seeds=torch.arange(first_seed,first_seed+num_seeds,1)


for seed in seeds:
    torch.manual_seed(seed)
    
    # load dataset
    dataset, n_classes, n_features, test_graph=get_dataset(dataset_name)

    n_nodes=len(dataset.x)

    if dataset_name=='Toy_graph_single':
       dataset_train,dataset_test=dataset
       dataset=dataset_train
    else: 
       dataset_test=dataset

    # init model
    if model_name=='LTFGW_GCN':
        model=LTFGW_GCN(n_classes=n_classes,n_features=n_features, n_templates=n_templates,n_templates_nodes=n_templates_nodes,hidden_layer=hidden_layer,train_node_weights=train_node_weights)

    elif model_name=='GCN_LTFGW':
        model=GCN_LTFGW(n_classes=n_classes,n_features=n_features, n_templates=6,n_templates_nodes=6, skip_connection=True)
  
    elif model_name=='MLP':
        model=MLP(n_classes=n_classes,n_features=n_features,hidden_layer=hidden_layer)

    elif model_name=='GCN':
        model=GCN(n_classes=n_classes,n_features=n_features,hidden_layer=hidden_layer,n_hidden_layers=n_hidden_layer)

    elif model_name=='LTFGW_MLP':
        model=LTFGW_MLP(n_nodes=n_nodes,n_classes=n_classes,n_features=n_features, n_templates=n_templates,n_templates_nodes=n_templates_nodes,hidden_layer=hidden_layer)


    method=model_name+'_'+training
    filename_save, filename_best_model, filename_visus = get_filenames(dataset_name,method,seed)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

    best_val_perf=0

 #   if training=='mini_batch':
 #       train_loader = NeighborLoader(dataset,num_neighbors= [-1],
 #       batch_size=8,
 #       input_nodes=dataset.train_mask,shuffle=True)
 #       train_minibatch(model,train_loader,dataset,optimizer,criterion,n_epoch,save,filename_save,filename_best_model,best_val_perf)
        
    if training=='complete_graph':
        generator = torch.Generator().manual_seed(seed.item())
        transform=T.RandomNodeSplit(split='random',num_val=int(0.2*len(dataset.x)),num_test=int(0.2*len(dataset.x)))
        dataset=transform(dataset)
        torch.save(dataset,'dataset')
        train(model,dataset,n_epoch,criterion, optimizer,save,filename_save,filename_best_model,best_val_perf,filename_visus)

    elif training=='multi_graph':
        generator = torch.Generator().manual_seed(seed.item())
        train_dataset,val_dataset,test_dataset=torch.utils.data.random_split(dataset,random_split,generator=generator)
        train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
        train_multi_graph(model,criterion,optimizer,n_epoch,save,filename_save,filename_best_model,train_loader,val_loader,filename_visus)

    checkpoint = torch.load(filename_best_model)
    model.load_state_dict(checkpoint['model_state_dict'])

    if training=='multi_graph':
      test_acc=test_multigraph(model,dataset_test)
      Test_accuracy.append(test_acc)

    elif training=='complete_graph':
        test_acc=test(model,dataset_test,test_graph)
        Test_accuracy.append(test_acc)    
    
    filename_save_test=os.path.join( 'results',method,"test_{}_seed{}.csv".format(dataset_name,first_seed))
    np.savetxt(filename_save_test,Test_accuracy)

#print the performances
Test_accuracy=torch.tensor(Test_accuracy)
print('mean_accuracy={},std={}'.format(torch.mean(Test_accuracy),torch.std(Test_accuracy)))



# %%
