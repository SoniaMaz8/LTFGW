import time
from tqdm import tqdm
import torch
import os 
import pandas as pd
from sklearn.manifold import TSNE
from utils import plot_TSNE


def train_epoch(dataset,model,criterion,optimizer,epoch):
    """"
    train one epoch on the complete graph
    """
    model.train()
    optimizer.zero_grad()  
    out = model(dataset.x,dataset.edge_index)
    if epoch%20==0:
       x_latent=out.detach().numpy()
       df_x=pd.DataFrame(x_latent)
       df_x.to_csv('results/TSNE2/latent{}.csv'.format(epoch))
    pred = out.argmax(dim=1)         # Use the class with highest probability
    train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]   #number of correct node predictions
    train_acc = int(train_correct.sum()) / int(dataset.train_mask.sum())  #training_accuracy
    val_correct=pred[dataset.val_mask] == dataset.y[dataset.val_mask]
    val_acc= int(val_correct.sum()) / int(dataset.val_mask.sum())
    loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  
    loss.backward()  
    optimizer.step()  
    return loss, train_acc, val_acc

def train_epoch_minibatch(model,train_loader,data,optimizer,criterion):
    """"
    train one epoch with minibatches
    """    
    model.train()
    total_loss = 0
    total_train_acc=0
    total_val_acc=0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        out = model(data.x,data.edge_index) 
        pred = out.argmax(dim=1)   #predcit the class with the highest probability
        train_correct = pred[data.train_mask] == data.y[data.train_mask]    #number of correct node predictions
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  #train accuracy
        total_train_acc+=train_acc
        val_correct=pred[data.val_mask] == data.y[data.val_mask]
        val_acc= int(val_correct.sum()) / int(data.val_mask.sum())  
        total_val_acc+=val_acc
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        total_loss += loss.item() 
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader), total_train_acc / len(train_loader),  total_val_acc / len(train_loader) #mean of train accuracy and loss over the mini batches


def train_minibatch(model,train_loader,dataset,optimizer,criterion,N_epoch,save,filename_save,filename_best_model,best_val_perf):
    """"
    train the entire model with minibatches
    """        
    Loss=[]
    Train_acc=[] 
    Val_acc=[]  
    if save:
      df=pd.DataFrame(columns=['loss','train_accuracy','validation_accuracy','test_accuracy','best_validation_accuracy'])           
    for epoch in range(N_epoch):  
        start=time.time()      
        loss,train_acc, val_acc = train_epoch_minibatch(model,train_loader,dataset,optimizer,criterion)
        Loss.append(loss)
        Train_acc.append(train_acc)  
        Val_acc.append(val_acc) 
        end=time.time()  
        if save: 
            df.at[len(df)-1,'loss']=Loss
            df.at[len(df)-1,'train_accuracy']=Train_acc
            df.at[len(df)-1,'validation_accuracy']=Val_acc
            if val_acc>best_val_perf:  
                torch.save(model.state_dict(),filename_best_model)
                best_val_perf=val_acc
                df.at[len(df)-1,'best_validation_accuracy']=val_acc
            df.to_pickle(filename_save) 
        print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f},Validation Accuracy:{val_acc:.4f}')
    if save:
        df = pd.read_pickle(filename_save)
        max_val=max(df.iloc[len(df)-1]['validation_accuracy'])
        df.at[len(df)-1,'max_val_accuracy']=max_val
        df.to_pickle(filename_save)  
    return Loss, Train_acc, Val_acc


def train(model,dataset,N_epoch,criterion, optimizer,save,filename_save,filename_best_model,best_val_perf):
    """"
    train the entire model on the entire graph
    """         
    Loss=[]
    Train_acc=[]
    Val_acc=[]
    if save:
      df=pd.DataFrame(columns=['loss','train_accuracy','validation_accuracy','test_accuracy','best_validation_accuracy'])    
    for epoch in tqdm(range(N_epoch)): 
            start=time.time()     
            loss,train_acc, val_acc = train_epoch(dataset,model,criterion,optimizer,epoch)
            end=time.time()
            Loss.append(loss.item())
            Train_acc.append(train_acc)
            Val_acc.append(val_acc)
            if save: 
                df.at[len(df)-1,'loss']=Loss
                df.at[len(df)-1,'train_accuracy']=Train_acc
                df.at[len(df)-1,'validation_accuracy']=Val_acc
                if val_acc>best_val_perf:  
                    torch.save(model.state_dict(),filename_best_model)
                    best_val_perf=val_acc
                    df.at[len(df)-1,'best_validation_accuracy']=val_acc
                df.to_pickle(filename_save)        
            print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f},Validation Accuracy:{val_acc:.4f}')  
    return Loss, Train_acc, Val_acc

def test(model,dataset):
    """"
    test the model
    """      
    model.eval()
    out = model(dataset.x,dataset.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]  
    test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum()) 
    return test_acc

