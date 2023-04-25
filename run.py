from torch_geometric.loader import NeighborLoader
from architectures import GCN_LTFGW, GCN_3_layers
import torch
from torch.optim.lr_scheduler import MultiStepLR
from data.convert_datasets import Citeseer_data
from trainers import train,train_minibatch
from tqdm import tqdm

torch.manual_seed(123456)

dataset_name='toy_graph'  #'citeseer' or 'toy_graph'
model_name='GCN'  #'LTFGW' or 'GCN'

if dataset_name=='citeseer':
    dataset=Citeseer_data()

    n_classes=6

    train_loader = NeighborLoader(dataset,num_neighbors= [-1],
    batch_size=8,
    input_nodes=dataset.train_mask,shuffle=True)
    
    if model_name=='LTFGW':
      model=GCN_LTFGW(n_classes=n_classes,N_features=dataset.num_features, N_templates=10,N_templates_nodes=10)
      #filename to save the loss, accuracies and model parameters
      filename_values='results/Citeseer_results.csv'
      filename_model='models/model_Citeseer.pt'
    else:
      model=GCN_3_layers(n_classes=n_classes,N_features=dataset.num_features,hidden_layer=20)
      #filename to save the loss, accuracies and model parameters
      filename_values='results/Citeseer_results_GCN.csv'
      filename_model='models/model_Citeseer_GCN.pt'

    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)    

    loss, train_acc=train_minibatch(model,train_loader,dataset,optimizer,criterion,50,False, filename_values,filename_model)

elif dataset_name=='toy_graph':
    dataset=torch.load('data/toy_graph1.pt')
    #filename to save the loss, accuracies and model parameters
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
        loss, train_acc=train(model,dataset,50,criterion,optimizer,False,filename_values,filename_model)
        Loss+=loss
        Train_acc+=train_acc
    print('mean loss={}'.format(Loss/num_seeds))
    print('mean train accuracy={}'.format(Train_acc/num_seeds))










