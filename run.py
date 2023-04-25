from torch_geometric.loader import NeighborLoader
from architectures import GCN_LTFGW, GCN_3_layers
import torch
from torch.optim.lr_scheduler import MultiStepLR
from data.convert_datasets import Citeseer_data
from train_Citeseer import train
from train_toy_graph import train_toy
from tqdm import tqdm

torch.manual_seed(123456)

dataset_name='citeseer'  #or 'toy_graph'

if dataset_name=='citeseer':
    dataset=Citeseer_data()
    filename_values='results/Citeseer_results.csv'
    filename_model='models/model_Citeseer.pt'
    n_classes=6

    train_loader = NeighborLoader(dataset,num_neighbors= [-1],
    batch_size=8,
    input_nodes=dataset.train_mask,shuffle=True)

    model=GCN_LTFGW(n_classes=n_classes,N_features=dataset.num_features, N_templates=10,N_templates_nodes=10)
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)    

    loss, train_acc=train(model,train_loader,dataset,optimizer,criterion,50,False, filename_values,filename_model)

if dataset_name=='toy_graph':
    dataset=torch.load('data/toy_graph1.pt')
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
        loss, train_acc=train_toy(model,dataset,50,criterion,optimizer,False)
        Loss+=loss
        Train_acc+=train_acc
    print('mean loss={}'.format(Loss/num_seeds))
    print('mean train accuracy={}'.format(Train_acc/num_seeds))










