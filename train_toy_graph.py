from architectures import GCN_LTFGW
import torch
from tqdm import tqdm
import csv
import datetime

dataset=torch.load('Toy_graphs/toy_graph1.pt')
model=GCN_LTFGW(n_classes=3,N_features=3)

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


def train_epoch(dataset, save=False):
      ''''
      save: bool, wether to save the parameters after each epoch or not
      '''
      model.train()
      optimizer.zero_grad()  
      out = model(dataset.x,dataset.edge_index) 
      pred = out.argmax(dim=1)         # Use the class with highest probability
      train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]   #number of correct node predictions
      train_acc = int(train_correct.sum()) / int(dataset.train_mask.sum())  #training_accuracy
      if save:

            
      loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  
      loss.backward()  
      optimizer.step()  
      return loss, train_acc


def train(dataset,N_epoch,save):
      ''''
      save: bool, wether to save the parameters after each epoch or not
      '''
      if save:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_column_name = f'Loss/train_accuracy_{now}'
            filename = 'loss_history.csv'
            with open(filename, 'a', newline='') as f:
                  writer = csv.writer(f)
                  writer.writerow([new_column_name]) 
      for epoch in range(N_epoch):      
            loss,train_acc = train_epoch(dataset)
            if save:
                  with open(filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch,loss.item(),train_acc])  
                        
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f}')     
      if save:
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'model_toy.pt')
            
train(dataset,50,True)
       


  
