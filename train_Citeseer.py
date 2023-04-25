from torch_geometric.loader import NeighborLoader
from architectures import GCN_LTFGW
import torch
import time
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import csv
import datetime

dataset=torch.load('Citeseer_data/graph_Citeseer.pt')

torch.manual_seed(123456)

train_loader = NeighborLoader(dataset,num_neighbors= [-1],
    batch_size=8,
    input_nodes=dataset.train_mask,shuffle=True)

model=GCN_LTFGW(n_classes=6,N_features=dataset.num_features, N_templates=10,N_templates_nodes=10)

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_epoch(dataset):
    model.train()
    total_loss = 0
    total_train_acc=0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        out = model(data.x,data.edge_index) 
        pred = out.argmax(dim=1) 
        train_correct = pred[data.train_mask] == data.y[data.train_mask]   
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
        total_train_acc+=train_acc
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        total_loss += loss.item() 
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader), total_train_acc / len(train_loader)

def train(dataset,N_epoch,save):
      ''''
      save: bool, wether to save the parameters after each epoch or not
      '''
      if save:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_column_name = f'Loss/train_accuracy_{now}'
            filename = 'results/Citeseer_results.csv'
            with open(filename, 'a', newline='') as f:
                  writer = csv.writer(f)
                  writer.writerow([new_column_name]) 
      for epoch in range(N_epoch):      
            loss,train_acc = train_epoch(dataset)
            if save:
                  with open(filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch,loss,train_acc])  
                        
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f}')     
      if save:
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'model_Citeseer.pt')
            
train(dataset,50,True)
