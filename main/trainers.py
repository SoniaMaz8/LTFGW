import time
from tqdm import tqdm
import torch
import os 
import pandas as pd
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


def train_epoch(dataset,model,criterion,optimizer):
    """"
    train one epoch on the complete graph
    """
    model.train()
    optimizer.zero_grad()  
    out,x_latent = model(dataset.x,dataset.edge_index)

    pred = out.argmax(dim=1)  

    #train    
    train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]   #number of correct node predictions
    train_acc = int(train_correct.sum()) / int(dataset.train_mask.sum())  #training_accuracy

    #validation
    val_correct=pred[dataset.val_mask] == dataset.y[dataset.val_mask]
    val_acc= int(val_correct.sum()) / int(dataset.val_mask.sum())

    loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
 
    loss_val=  criterion(out[dataset.val_mask], dataset.y[dataset.val_mask])
    loss.backward()  
    optimizer.step()  

    return loss, train_acc, val_acc, x_latent,loss_val

def train(model,dataset,N_epoch,criterion, optimizer,save,filename_save,filename_best_model,filename_visus):
    """"
    train the entire model on the entire graph
    """     

    best_val_perf=0  

    if save:
      #create dataframe to save performances
      df=pd.DataFrame(columns=['loss','loss_validation','train_accuracy','validation_accuracy','test_accuracy','best_validation_accuracy']) 
      df.to_pickle(filename_save)

    for epoch in tqdm(range(N_epoch)): 
            start=time.time()     
            loss,train_acc, val_acc, x_latent,loss_val = train_epoch(dataset,model,criterion,optimizer)
            end=time.time()

            if save: 
                df=pd.read_pickle(filename_save)
                #add performances to the dataframe
                df.at[epoch,'loss']=loss.item()
                df.at[epoch,'train_accuracy']=train_acc
                df.at[epoch,'validation_accuracy']=val_acc
                df.at[epoch,'loss_validation']=loss_val.item()

                if val_acc>best_val_perf:  

                    #save best model parameters
                    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},filename_best_model)
                    best_val_perf=val_acc
                    df.at[epoch,'best_validation_accuracy']=val_acc

                    #save latent embedding for visualisation
                    x_latent=x_latent.detach().numpy()
                    df_x=pd.DataFrame(x_latent)
                    df_x.to_csv(filename_visus)

                df.to_pickle(filename_save) 

            #print performances           
            print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Loss validation: {loss_val:.4f},Train Accuracy: {train_acc:.4f},Validation Accuracy:{val_acc:.4f}')  


def test(model,dataset,test_graph):
    """"
    test the model
    test_graph: True if the test is done on the whole dataset, False if it is a subset of the dataset
    """      
    model.eval()
    out,_= model(dataset.x,dataset.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    if test_graph:
      test_correct = pred == dataset.y
    else:
      test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]
    test_acc = int(test_correct.sum())/ len(dataset.x[dataset.test_mask])
    return test_acc


def train_epoch_multi_graph(model,criterion,optimizer,train_loader):

    model.train()
    optimizer.zero_grad()  
    Loss=[]
    Train_acc=[]
    X_latent=[]
    Data=[]
    for data in train_loader:  # Iterate in batches over the training dataset.
           Data.append(data) #save data for visualisation

           out,x_latent = model(data.x, data.edge_index)  # Perform a single forward pass.
    
           pred=out.argmax(dim=1) 

           train_correct = pred == data.y   #number of correct node predictions
           train_acc = int(train_correct.sum()) / int(len( data.y))  #training_accuracy

           loss = criterion(out, data.y)

           loss.backward()  
           optimizer.step() 
           optimizer.zero_grad() 

           Loss.append(loss)
           Train_acc.append(train_acc)
           X_latent.append(x_latent.detach().numpy())

    Loss=torch.Tensor(Loss)
    Train_acc=torch.Tensor(Train_acc)

    return torch.mean(Loss), torch.mean(Train_acc), X_latent, Data


def validation_epoch_multi_graph(model,val_loader,criterion):
    model.eval() 
    Val_acc=[]
    X_latent=[]
    Data=[]
    Loss_val=[]

    for data in val_loader:  # Iterate in batches over the training dataset.
           Data.append(data)  #save data for visualisation
           out,x_latent = model(data.x, data.edge_index)  # Perform a single forward pass.
           pred=out.argmax(dim=1) 
           Loss_val.append( criterion(out, data.y))
           val_correct = pred == data.y   #number of correct node predictions
           val_acc = int(val_correct.sum()) / int(len( data.y))  #training_accuracy
           Val_acc.append(val_acc)
           X_latent.append(x_latent.detach().numpy())

    Loss_val=torch.Tensor(Loss_val)
    Val_acc=torch.Tensor(Val_acc)

    return torch.mean(Val_acc),X_latent, Data,torch.mean(Loss_val)



def train_multi_graph(model,criterion,optimizer,n_epoch,save,filename_save,filename_best_model,train_loader,val_loader,filename_visus):

    best_val_perf=0
    if save:
      #create dataframe to save the performances
      df=pd.DataFrame(columns=['loss','loss_validation','train_accuracy','validation_accuracy','test_accuracy','best_validation_accuracy']) 
      df.to_pickle(filename_save)  

    for epoch in range(n_epoch):
      
      #training
      start=time.time()
      loss,train_acc, x_latent,Data=train_epoch_multi_graph(model,criterion,optimizer,train_loader)
      end=time.time()

      #validation
      val_acc,x_latent_val,Data_validation,loss_val=validation_epoch_multi_graph(model,val_loader,criterion)

      print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Loss validation:{loss_val:.4f},Train Accuracy: {train_acc:.4f},Validation Accuracy:{val_acc:.4f}') 
      
      if save: 
        #add the performances to the dataframe
        df.at[epoch,'loss']=loss.item()
        df.at[epoch,'train_accuracy']=train_acc
        df.at[epoch,'validation_accuracy']=val_acc
        df.at[epoch,'loss_validation']=loss_val

        if val_acc>best_val_perf:
            #save best model parameters 
            torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},filename_best_model)
            best_val_perf=val_acc
            df.at[epoch,'best_validation_accuracy']=val_acc 
            
            #save latent embedding for visualisation: train and validation data
            df_x=pd.DataFrame(x_latent)
            df_x.to_pickle(filename_visus)
            Data=pd.DataFrame(Data)
            Data.to_pickle('Data_'+filename_visus)

            x_latent_val=x_latent_val
            df_x_val=pd.DataFrame(x_latent_val)
            df_x_val.to_pickle('validation_'+filename_visus)
            Data_validation=pd.DataFrame(Data_validation)
            Data_validation.to_pickle('Data_'+filename_visus)
            
            
        df.to_pickle(filename_save)

def test_multigraph(model,dataset):
    """"
    test the model
    """      
    model.eval()
    Test_acc=[]
    for data in dataset:
        out,_= model(data.x,data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred == data.y
        test_acc = int(test_correct.sum())/ len(data.x)
        Test_acc.append(test_acc)
    Test_acc=torch.tensor(Test_acc)
    return  torch.mean(Test_acc) 

def train_epoch_minibatch(data,criterion,optimizer,model,loader):
    model.train()
    total_loss =[]
    total_train_acc=0
    for data in loader:
      
        optimizer.zero_grad()
        out,_ = model(data.x,data.edge_index) 
        pred = out.argmax(dim=1) 
        train_correct = pred[data.train_mask] == data.y[data.train_mask]   
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
        total_train_acc+=train_acc
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        total_loss.append(loss)

    mean_loss = torch.mean(torch.stack(total_loss))
    mean_loss.backward()
    optimizer.step()
      

    return  mean_loss, total_train_acc / len(loader)

def validation_epoch_minibatch(model,loader,criterion):
    """"
    test the model
    test_graph: True if the test is done on the whole dataset, False if it is a subset of the dataset
    """      
    model.eval()
    total_val_acc=0
    total_loss_val=0
    for data in loader:
       out,_= model(data.x,data.edge_index)
       pred = out.argmax(dim=1)  # Use the class with highest probability.
       val_correct = pred[data.val_mask] == data.y[data.val_mask]   
       val_acc = int(val_correct.sum()) / int(data.val_mask.sum())  
       total_val_acc+=val_acc
       loss_val=criterion(out[data.val_mask], data.y[data.val_mask])
       total_loss_val+=loss_val

    return val_acc/len(loader),total_loss_val/len(loader)


            
def train_minibatch(model,dataset,N_epoch,criterion, optimizer,save,filename_save,filename_best_model,filename_visus,loader,loader_val):
    """"
    train the entire model on the entire graph
    """     

    best_val_perf=0  

    if save:
      #create dataframe to save performances
      df=pd.DataFrame(columns=['loss','loss_validation','train_accuracy','validation_accuracy','test_accuracy','best_validation_accuracy']) 
      df.to_pickle(filename_save)

    for epoch in tqdm(range(N_epoch)): 
            start=time.time()     
            loss,train_acc =train_epoch_minibatch(dataset,criterion,optimizer,model,loader)
            val_acc,loss_val =validation_epoch_minibatch(model,loader_val,criterion)
            end=time.time()

            if save: 
                df=pd.read_pickle(filename_save)
                #add performances to the dataframe
                df.at[epoch,'loss']=loss.item()
                df.at[epoch,'train_accuracy']=train_acc
                df.at[epoch,'validation_accuracy']=val_acc
                df.at[epoch,'loss_validation']=loss_val.item()

                if val_acc>best_val_perf:  

                    #save best model parameters
                    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},filename_best_model)
                    best_val_perf=val_acc
                    df.at[epoch,'best_validation_accuracy']=val_acc

                    #save latent embedding for visualisation
  #                  x_latent=x_latent.detach().numpy()
  #                  df_x=pd.DataFrame(x_latent)
  #                  df_x.to_csv(filename_visus)

                df.to_pickle(filename_save) 

            #print performances           
            print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Loss validation: {loss_val:.4f},Train Accuracy: {train_acc:.4f},Validation Accuracy:{val_acc:.4f}') 



def test_minibatch(model,loader):
    """"
    test the model
    test_graph: True if the test is done on the whole dataset, False if it is a subset of the dataset
    """      
    model.eval()
    total_test_acc=0
    for data in tqdm(loader):
       out,_= model(data.x,data.edge_index)
       pred = out.argmax(dim=1)  # Use the class with highest probability.
       test_correct = pred[data.test_mask] == data.y[data.test_mask]   
       test_acc = int(test_correct.sum()) / int(data.train_mask.sum())  
       total_test_acc+=test_acc
    return test_acc/len(loader)