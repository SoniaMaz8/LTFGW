import time
from tqdm import tqdm
import torch
import os
import pandas as pd
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import StepLR


def train_epoch(dataset, model, criterion, optimizer):

    model.train()
    optimizer.zero_grad()
    out = model(dataset.x, dataset.edge_index)

    pred = out.argmax(dim=1)

    # train
    # number of correct node predictions
    train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]
    train_acc = int(train_correct.sum()) / \
        int(dataset.train_mask.sum())  # training_accuracy
    
    
    loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss, train_acc


def val_epoch(dataset, model, criterion):

    model.eval()
    out, _ = model(dataset.x, dataset.edge_index)

    pred = out.argmax(dim=1)

    val_correct = pred[dataset.val_mask] == dataset.y[dataset.val_mask]
    val_acc = int(val_correct.sum()) / int(dataset.val_mask.sum())

    loss_val = criterion(out[dataset.val_mask], dataset.y[dataset.val_mask])

    return loss_val, val_acc


def train(criterion,optimizer,loader,loader_val,model,filename_save,filename_best_model,filename_visus,filename_templates,filename_alpha,filename_current_model,save,schedule, template_sizes,nepochs,model_name,dataset,loader_test):

    best_val_perf = 0
    Templates = []
    alphas = []
    if schedule:
        scheduler = StepLR(optimizer, 200, 0.8)

    if save:
      save_templates = model_name in ['LTFGW_MLP','LTFGW_GCN','LTFGW_MLP_log','LTFGW_MLP_dropout','LTFGW_MLP_semirelaxed','LTFGW_MLP_dropout','LTFGW_MLP_dropout_relu']
    
    else: 
        save_templates=False

    if save:
        # create dataframe to save performances
        df = pd.DataFrame(
            columns=[
                'loss',
                'loss_validation',
                'train_accuracy',
                'validation_accuracy',
                'test_accuracy',
                'best_validation_accuracy'])
    
    for epoch in tqdm(range(nepochs)):

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        start = time.time()
        
        for dataset in loader:
            loss, train_acc = train_epoch(
                dataset, model, criterion, optimizer)
            train_losses.append(loss.item())
            train_accs.append(train_acc)

        for dataset in loader_val:
            loss_val, val_acc = val_epoch(dataset, model, criterion)
            val_losses.append(loss_val.item())
            val_accs.append(val_acc)


        mean_train_acc = torch.mean(torch.tensor(train_accs))
        mean_train_loss = torch.mean(torch.tensor(train_losses))
        mean_val_acc = torch.mean(torch.tensor(val_accs))
        mean_val_loss = torch.mean(torch.tensor(val_losses))

        if save:
            # add performances to the dataframe
            df.at[epoch, 'loss'] = mean_train_loss
            df.at[epoch, 'train_accuracy'] = mean_train_acc
            df.at[epoch, 'validation_accuracy'] = mean_val_acc
            df.at[epoch, 'loss_validation'] = mean_val_loss

            if  mean_val_acc >= best_val_perf:

                # save best model parameters
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           filename_best_model)
                best_val_perf = mean_val_acc
                df.at[epoch, 'best_validation_accuracy'] = mean_val_acc

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       filename_current_model)

            if save_templates and template_sizes==None:

                df_model = torch.load(filename_current_model)
                Templates.append(df_model['model_state_dict']['LTFGW.templates'])
                alphas.append(df_model['model_state_dict']['LTFGW.alpha0'])

        end = time.time()

        checkpoint = torch.load(filename_best_model)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_acc = test(model, loader_test)
        print(test_acc)

        checkpoint = torch.load(filename_current_model)
        model.load_state_dict(checkpoint['model_state_dict'])

        if save:
          df.to_pickle(filename_save)
        # print performances
        print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {mean_train_loss:.4f},Loss validation: {mean_val_loss:.4f},Train Accuracy: {mean_train_acc:.4f},Validation Accuracy:{mean_val_acc:.4f}')

        if save_templates and template_sizes==None:
            torch.save(Templates, filename_templates)
            torch.save(alphas, filename_alpha)

        if schedule:
            scheduler.step()         

def test(model, loader):
    """"
    test the model
    test_graph: True if the test is done on the whole dataset, False if it is a subset of the dataset
    """
    model.eval()
    test_accs=[]
    for data in loader:
        out= model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_accs.append(int(test_correct.sum()) / len(data.x[data.test_mask]))
    return torch.mean(torch.tensor(test_accs))


def train_epoch_multi_graph(model, criterion, optimizer, train_loader):

    model.train()
    optimizer.zero_grad()
    Loss = []
    Train_acc = []
    Data = []
    for data in train_loader:  # Iterate in batches over the training dataset.
        Data.append(data)  # save data for visualisation

        # Perform a single forward pass.
        out = model(data.x, data.edge_index, data.batch)

        pred = torch.argmax(out,dim=1)


        train_correct = pred == data.y  # number of correct node predictions
        train_acc = train_correct.sum()/len(data)
        
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        Loss.append(loss)
        Train_acc.append(train_acc)

    
    Loss = torch.tensor(Loss)
    Train_acc = torch.tensor(Train_acc)

    return torch.mean(Loss), torch.mean(Train_acc), Data


def validation_epoch_multi_graph(model, val_loader, criterion):
    model.eval()
    Val_acc = []
    Data = []
    Loss_val = []

    for data in val_loader:  # Iterate in batches over the training dataset.
        Data.append(data)  # save data for visualisation
        # Perform a single forward pass.
        out = model(data.x, data.edge_index,data.batch)
        pred = torch.argmax(out,dim=1)
        Loss_val.append(criterion(out, data.y))
        val_correct = pred == data.y  # number of correct node predictions
        val_acc = val_correct.sum()/len(data)
        Val_acc.append(val_acc)


    Loss_val = torch.Tensor(Loss_val)
    Val_acc = torch.Tensor(Val_acc)

    return torch.mean(Val_acc), Data, torch.mean(Loss_val)


def train_multi_graph(
        model,
        criterion,
        optimizer,
        n_epoch,
        save,
        filename_save,
        filename_best_model,
        train_loader,
        val_loader,
        filename_visus):

    best_val_perf = 0
    if save:
        # create dataframe to save the performances
        df = pd.DataFrame(
            columns=[
                'loss',
                'loss_validation',
                'train_accuracy',
                'validation_accuracy',
                'test_accuracy',
                'best_validation_accuracy'])
        df.to_pickle(filename_save)

    for epoch in range(n_epoch):

        # training
        start = time.time()
        loss, train_acc, Data = train_epoch_multi_graph(
            model, criterion, optimizer, train_loader)
        end = time.time()

        # validation
        val_acc, Data_validation, loss_val = validation_epoch_multi_graph(
            model, val_loader, criterion)

        print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Loss validation:{loss_val:.4f},Train Accuracy: {train_acc:.4f},Validation Accuracy:{val_acc:.4f}')

        if save:
            # add the performances to the dataframe
            df.at[epoch, 'loss'] = loss.item()
            df.at[epoch, 'train_accuracy'] = train_acc
            df.at[epoch, 'validation_accuracy'] = val_acc
            df.at[epoch, 'loss_validation'] = loss_val

            if val_acc > best_val_perf:
                # save best model parameters
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           filename_best_model)
                best_val_perf = val_acc
                df.at[epoch, 'best_validation_accuracy'] = val_acc
            df.to_pickle(filename_save)


def test_multigraph(model, dataset):
    """"
    test the model
    """
    model.eval()
    Test_acc = []
    for data in dataset:
        out= model(data.x, data.edge_index)
        pred = out.argmax
        test_correct = pred == data.y
        test_acc = test_correct.sum()/len(data)
        Test_acc.append(test_acc)
    Test_acc = torch.tensor(Test_acc)
    return torch.mean(Test_acc)

def train2(criterion,optimizer,loader,loader_val,model,filename_save,filename_best_model,filename_visus,filename_templates,filename_alpha,filename_current_model,save,schedule, template_sizes,nepochs,model_name):

    best_val_perf = 0
    Templates = []
    alphas = []
    if schedule:
        scheduler = StepLR(optimizer, 200, 0.8)

    if save:
      save_templates = model_name in ['LTFGW_MLP','LTFGW_GCN','LTFGW_MLP_log','LTFGW_MLP_dropout','LTFGW_MLP_semirelaxed','LTFGW_MLP_dropout','LTFGW_MLP_dropout_relu']
    
    else: 
        save_templates=False

    if save:
        # create dataframe to save performances
        df = pd.DataFrame(
            columns=[
                'loss',
                'loss_validation',
                'train_accuracy',
                'validation_accuracy',
                'test_accuracy',
                'best_validation_accuracy'])
    
    for epoch in tqdm(range(nepochs)):

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        start = time.time()
        
        for dataset in loader:
            loss, train_acc = train_epoch(
                dataset, model, criterion, optimizer)
            train_losses.append(loss.item())
            train_accs.append(train_acc)

        for dataset in loader_val:
            loss_val, val_acc = val_epoch(dataset, model, criterion)
            val_losses.append(loss_val.item())
            val_accs.append(val_acc)


        mean_train_acc = torch.mean(torch.tensor(train_accs))
        mean_train_loss = torch.mean(torch.tensor(train_losses))
        mean_val_acc = torch.mean(torch.tensor(val_accs))
        mean_val_loss = torch.mean(torch.tensor(val_losses))

        if save:
            # add performances to the dataframe
            df.at[epoch, 'loss'] = mean_train_loss
            df.at[epoch, 'train_accuracy'] = mean_train_acc
            df.at[epoch, 'validation_accuracy'] = mean_val_acc
            df.at[epoch, 'loss_validation'] = mean_val_loss

            if  mean_val_acc >= best_val_perf:

                # save best model parameters
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           filename_best_model)
                best_val_perf = mean_val_acc
                df.at[epoch, 'best_validation_accuracy'] = mean_val_acc
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       filename_current_model)

            if save_templates and template_sizes==None:

                df_model = torch.load(filename_current_model)
                Templates.append(df_model['model_state_dict']['LTFGW.templates'])
                alphas.append(df_model['model_state_dict']['LTFGW.alpha0'])

        end = time.time()

        if save:
          df.to_pickle(filename_save)
        # print performances
        print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {mean_train_loss:.4f},Loss validation: {mean_val_loss:.4f},Train Accuracy: {mean_train_acc:.4f},Validation Accuracy:{mean_val_acc:.4f}')

        if save_templates and template_sizes==None:
            torch.save(Templates, filename_templates)
            torch.save(alphas, filename_alpha)

        if schedule:
            scheduler.step()   
