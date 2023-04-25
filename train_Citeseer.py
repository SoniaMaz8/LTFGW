from torch_geometric.loader import NeighborLoader
from architectures import GCN_LTFGW, GCN_3_layers
import torch
import time
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import csv
import datetime
from data.convert_datasets import Citeseer_data

torch.manual_seed(123456)

dataset=Citeseer_data()

train_loader = NeighborLoader(dataset,num_neighbors= [-1],
    batch_size=8,
    input_nodes=dataset.train_mask,shuffle=True)

model=GCN_LTFGW(n_classes=6,N_features=dataset.num_features, N_templates=10,N_templates_nodes=10)
#model=GCN_3_layers(n_classes=6,N_features=dataset.num_features)

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_epoch(dataset):
    model.train()
    total_loss = 0
    total_train_acc=0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        out = model(data.x,data.edge_index) 
        pred = out.argmax(dim=1)   #predcit the class with the highest probability
        train_correct = pred[data.train_mask] == data.y[data.train_mask]    #number of correct node predictions
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  #train accuracy
        total_train_acc+=train_acc
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        total_loss += loss.item() 
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader), total_train_acc / len(train_loader)  #mean of train accuracy and loss over the mini batches

def train(dataset,N_epoch,save,filename_values,filename_model):
      ''''
      save: bool, wether to save the parameters after each epoch or not
      filename_values: filename to store the loss and train accuracies
      flename_model: filename to store the model parameters
      '''
      if save:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_column_name = f'Loss/train_accuracy_{now}'
            filename = filename_values
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
      torch.save(model.state_dict(),filename_model)
      return loss, train_acc
            

