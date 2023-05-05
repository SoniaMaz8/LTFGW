import time
from tqdm import tqdm
import torch
import os 
import pandas as pd
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader



def train_epoch(dataset,model,criterion,optimizer,epoch):
    """"
    train one epoch on the complete graph
    """
    model.train()
    optimizer.zero_grad()  
    out,x_latent = model(dataset.x,dataset.edge_index)

    #save for visualisation
    if epoch%20==0:
       x_latent=x_latent.detach().numpy()
       df_x=pd.DataFrame(x_latent)
       df_x.to_csv('results/TSNE_{}/latent{}.csv'.format(str(model),epoch))

    pred = out.argmax(dim=1)   

    #train    
    train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]   #number of correct node predictions
    train_acc = int(train_correct.sum()) / int(dataset.train_mask.sum())  #training_accuracy

    #validation
    val_correct=pred[dataset.val_mask] == dataset.y[dataset.val_mask]
    val_acc= int(val_correct.sum()) / int(dataset.val_mask.sum())

    loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  
    loss.backward()  
    optimizer.step()  
    return loss, train_acc, val_acc

def train(model,dataset,N_epoch,criterion, optimizer,save,filename_save,filename_best_model,best_val_perf):
    """"
    train the entire model on the entire graph
    """         

    if save:
      df=pd.DataFrame(columns=['loss','train_accuracy','validation_accuracy','test_accuracy','best_validation_accuracy']) 
      df.to_pickle(filename_save)    
    for epoch in tqdm(range(N_epoch)): 
            start=time.time()     
            loss,train_acc, val_acc = train_epoch(dataset,model,criterion,optimizer,epoch)
            end=time.time()
            df=pd.read_pickle(filename_save)
            if save: 
                df.at[epoch,'loss']=loss.item()
                df.at[epoch,'train_accuracy']=train_acc
                df.at[epoch,'validation_accuracy']=val_acc
                if val_acc>best_val_perf:  
                    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},filename_best_model)
                    best_val_perf=val_acc
                    df.at[epoch,'best_validation_accuracy']=val_acc
                df.to_pickle(filename_save)        
            print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f},Validation Accuracy:{val_acc:.4f}')  


def test(model,dataset):
    """"
    test the model
    """      
    model.eval()
    out= model(dataset.x,dataset.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred == dataset.y
    test_acc = int(test_correct.sum())/ 1000
    return test_acc


def train_epoch_multi_graph(model,criterion,optimizer,train_loader):

    model.train()
    optimizer.zero_grad()  
    Loss=[]
    Train_acc=[]
    for data in train_loader:  # Iterate in batches over the training dataset.
           out,_ = model(data.x, data.edge_index)  # Perform a single forward pass.
           pred=out.argmax(dim=1) 

           train_correct = pred == data.y   #number of correct node predictions
           train_acc = int(train_correct.sum()) / int(len( data.y))  #training_accuracy
            
           loss = criterion(out, data.y)

           loss.backward()  
           optimizer.step() 
           optimizer.zero_grad() 

           Loss.append(loss)
           Train_acc.append(train_acc)

    Loss=torch.Tensor(Loss)
    Train_acc=torch.Tensor(Train_acc)

    return torch.mean(Loss), torch.mean(Train_acc)


def validation_epoch_multi_graph(model,val_loader):
    model.eval() 
    Train_acc=[]
    for data in val_loader:  # Iterate in batches over the training dataset.
           out,_ = model(data.x, data.edge_index)  # Perform a single forward pass.
           pred=out.argmax(dim=1) 

           val_correct = pred == data.y   #number of correct node predictions
           val_acc = int(val_correct.sum()) / int(len( data.y))  #training_accuracy
           Train_acc.append(val_acc)

    Train_acc=torch.Tensor(Train_acc)

    return torch.mean(Train_acc)



def train_multi_graph(model,criterion,optimizer,n_epoch,save,filename_save,filename_best_model,train_loader,val_loader):

    best_val_perf=0
    if save:
      df=pd.DataFrame(columns=['loss','train_accuracy','validation_accuracy','test_accuracy','best_validation_accuracy']) 
      df.to_pickle(filename_save)  

    for epoch in range(n_epoch):
      
      #training
      start=time.time()
      loss,train_acc=train_epoch_multi_graph(model,criterion,optimizer,train_loader)
      end=time.time()

      #validation
      val_acc=validation_epoch_multi_graph(model,val_loader)

      print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f},Validation Accuracy:{val_acc:.4f}') 

      if save: 
        df.at[epoch,'loss']=loss.item()
        df.at[epoch,'train_accuracy']=train_acc
        df.at[epoch,'validation_accuracy']=val_acc
        if val_acc>best_val_perf:  
            torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},filename_best_model)
            best_val_perf=val_acc
            df.at[epoch,'best_validation_accuracy']=val_acc
            df.to_pickle(filename_save)        
       

    
