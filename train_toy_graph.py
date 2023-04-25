from architectures import GCN_LTFGW
import torch
import csv
import datetime
from tqdm import tqdm
import time 

def train_epoch(dataset,model,criterion,optimizer):
      model.train()
      optimizer.zero_grad()  
      out = model(dataset.x,dataset.edge_index)
      pred = out.argmax(dim=1)         # Use the class with highest probability
      train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]   #number of correct node predictions
      train_acc = int(train_correct.sum()) / int(dataset.train_mask.sum())  #training_accuracy
      loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  
      loss.backward()  
      optimizer.step()  
      return loss, train_acc

def train_toy(model,dataset,N_epoch,criterion, optimizer,save):
      ''''
      save: bool, wether to save the parameters after each epoch or not
      '''
      if save:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_column_name = f'Loss/train_accuracy_{now}'
            filename = 'results/toy_results.csv'
            with open(filename, 'a', newline='') as f:
                  writer = csv.writer(f)
                  writer.writerow([new_column_name]) 
      for epoch in range(N_epoch): 
            start=time.time()     
            loss,train_acc = train_epoch(dataset,model,criterion,optimizer)
            end=time.time()
            if save:
                  with open(filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch,loss.item(),train_acc])  
            print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f}')     
      torch.save( model.state_dict(), 'models/model_toy.pt')
      return loss, train_acc
            

dataset=torch.load('data/toy_graph1.pt')
  

       


  
