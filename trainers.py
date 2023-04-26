import torch
import time
from tqdm import tqdm
import csv
import datetime
import torch.nn.functional as F
from architectures import GCN_3_layers
import pandas as pd

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

def train_epoch_minibatch(model,train_loader,data,optimizer,criterion):
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


def train_minibatch(model,train_loader,dataset,optimizer,criterion,N_epoch,save,filename_values,filename_model):
      ''''
      save: bool, wether to save the parameters after each epoch or not
      filename_values: filename to store the loss and train accuracies
      flename_model: filename to store the model parameters
      '''
      Loss=[]
      Train_acc=[]      
      for epoch in range(N_epoch):  
            start=time.time()      
            loss,train_acc = train_epoch_minibatch(model,train_loader,dataset,optimizer,criterion)
            Loss.append(loss.item())
            Train_acc.append(train_acc)            
            end=time.time()   
            print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f}')     
      return Loss, Train_acc


def train(model,dataset,N_epoch,criterion, optimizer,save,filename_values,filename_model):
      ''''
      save: bool, wether to save the parameters after each epoch or not
      '''
      Loss=[]
      Train_acc=[]
      for epoch in range(N_epoch): 
            start=time.time()     
            loss,train_acc = train_epoch(dataset,model,criterion,optimizer)
            end=time.time()
            Loss.append(loss.item())
            Train_acc.append(train_acc)
            print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f}')     
      return Loss, Train_acc

def test(model,dataset):
      model.eval()
      out = model(dataset.x,dataset.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]  
      test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum()) 
      return test_acc

