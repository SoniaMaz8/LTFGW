import time
from tqdm import tqdm
import torch
import pandas as pd
import os

def train_epoch(dataset,model,criterion,optimizer):
    """"
    train one epoch on the complete graph
    """
    model.train()
    optimizer.zero_grad()  
    out = model(dataset.x,dataset.edge_index)
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


def train_minibatch(model,train_loader,dataset,optimizer,criterion,N_epoch,save,best_val_perf,seed,dataset_name,model_name):
    """"
    train the entire model with minibatches
    """        
    Loss=[]
    Train_acc=[] 
    Val_acc=[] 
    df = pd.read_pkl('results/performances.pkl')
    df=df[model_name+'_minibatch',dataset_name]
    best_val_perf=max(df['validation_accuracy'])        
    for epoch in range(N_epoch):  
        start=time.time()      
        loss,train_acc, val_acc = train_epoch_minibatch(model,train_loader,dataset,optimizer,criterion)
        if save:
            df = pd.read_pkl('results/performances.pkl')
            df.at[len(df)-1,'model']=model_name+'_minibatch'
            df.at[len(df)-1,'dataset']=dataset_name
            df.at[len(df)-1,'loss']=loss
            df.at[len(df)-1,'seed']=seed
            df.at[len(df)-1,'train_accuracy']=train_acc
            df.at[len(df)-1,'validation_accuracy']=val_acc
            df.to_pkl('results/performances.pkl') 
            if val_acc> best_val_perf:
                    df_model=pd.DataFrame(columns=['seed','model_parameters'])
                    df_model['seed']=seed
                    df_model['model_parameters']=model.state_dict()
                    df_model.to_pkl(os.path.join('results',str(model_name)+'_'+str(dataset_name)+'.pkl'))
            df.to_pkl('results/performances.pkl') 
        Loss.append(loss)
        Train_acc.append(train_acc)  
        Val_acc.append(val_acc) 
        end=time.time()  
        if save: 
            if val_acc>best_val_perf:  
              filename_best_model=os.path.join( 'results',str(model),str(dataset_name)+ '.pkl')
              torch.save(model.state_dict(),filename_best_model)
        print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f},Validation Accuracy:{val_acc:.4f}') 
    return Loss, Train_acc, Val_acc


def train(model,dataset,N_epoch,criterion, optimizer,save,filename_save,filename_best_model,seed,dataset_name,model_name):
    """"
    train the entire model on the entire graph
    """         
    Loss=[]
    Train_acc=[]
    Val_acc=[] 
    df = pd.read_pkl('results/performances.pkl')
    df=df[str(model_name),str(dataset_name)]
    best_val_perf=max(df['validation_accuracy'])
    for epoch in tqdm(range(N_epoch)): 
            start=time.time()     
            loss,train_acc, val_acc = train_epoch(dataset,model,criterion,optimizer,dataset_name,model_name,seed)
            end=time.time()
            if save:
                df = pd.read_pkl('results/performances.pkl')
                df.at[len(df)-1,'model']=model_name
                df.at[len(df)-1,'dataset']=dataset_name
                df.at[len(df)-1,'loss']=loss
                df.at[len(df)-1,'seed']=seed
                df.at[len(df)-1,'train_accuracy']=train_acc
                df.at[len(df)-1,'validation_accuracy']=val_acc
                if val_acc> best_val_perf:
                    df_model=pd.DataFrame(columns=['seed','model_parameters'])
                    df_model['seed']=seed
                    df_model['model_parameters']=model.state_dict()
                    df_model.to_pkl(os.path.join('results',str(model_name)+'_'+str(dataset_name)+'.pkl'))
                df.to_pkl('results/performances.pkl') 
            Loss.append(loss.item())
            Train_acc.append(train_acc)
            Val_acc.append(val_acc)    
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


