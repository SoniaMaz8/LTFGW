from torch_geometric.loader import NeighborLoader
from architectures import GCN_LTFGW, GCN_3_layers
import torch
from torch.optim.lr_scheduler import MultiStepLR
from data.convert_datasets import Citeseer_data
from trainers import train,train_minibatch, train_multiple_seeds, test
from tqdm import tqdm
import torch.nn.functional as F

#torch.manual_seed(123456)

dataset_name='citeseer'  #'citeseer' or 'toy_graph'
model_name='GCN'  #'LTFGW' or 'GCN'
save_parameters=True
N_epoch=200


if dataset_name=='citeseer':
    dataset=Citeseer_data()

    n_classes=6

    train_loader = NeighborLoader(dataset,num_neighbors= [-1],
    batch_size=8,
    input_nodes=dataset.train_mask,shuffle=True)
    
    if model_name=='LTFGW':
      model=GCN_LTFGW(n_classes=n_classes,N_features=dataset.num_features, N_templates=10,N_templates_nodes=10)
      #filenames to save the loss, accuracies and model parameters
      filename_values='results/Citeseer_results.csv'
      filename_model='models/model_Citeseer.pt'
      criterion = torch.nn.CrossEntropyLoss() 
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=5e-4) 
      loss, train_acc=train_minibatch(model,train_loader,dataset,optimizer,criterion,200,save_parameters, filename_values,filename_model)

    else:
      model=GCN_3_layers(n_classes=n_classes,N_features=dataset.num_features,dropout=0.6)
      #filenames to save the loss, accuracies and model parameters
      filename_values='results/Citeseer_results_GCN.csv'
      filename_model='models/model_Citeseer_GCN.pt'
      criterion = torch.nn.CrossEntropyLoss() 
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=5e-4) 
      #train(model,dataset,N_epoch,criterion, optimizer,False,filename_values,filename_model)
      #mean_test_acc=test(model,dataset)
      mean_test_acc=train_multiple_seeds(model,dataset,N_epoch,save_parameters,filename_values,filename_model,n_classes,dataset.num_features,criterion,optimizer)
      print(mean_test_acc)

elif dataset_name=='toy_graph':
    dataset=torch.load('data/toy_graph1.pt')
    #filenames to save the loss, accuracies and model parameters
    filename_values='results/toy_results.csv'
    filename_model='models/model_toy.pt'  
    n_classes=3

    Loss=0
    Train_acc=0
    num_seeds=10
    seeds=torch.randint(100,(num_seeds,))
    for seed in tqdm(seeds):
        torch.manual_seed(seed)
        model=GCN_LTFGW(n_classes=3,N_features=3)
        criterion = torch.nn.CrossEntropyLoss()  
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss, train_acc=train(model,dataset,N_epoch,criterion,optimizer,save_parameters,filename_values,filename_model)
        Loss+=loss
        Train_acc+=train_acc
    print('mean loss={}'.format(Loss/num_seeds))
    print('mean train accuracy={}'.format(Train_acc/num_seeds))










