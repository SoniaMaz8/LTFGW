import time
from tqdm import tqdm
import torch
import os
import pandas as pd
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import is_undirected


def train_epoch(dataset, model, criterion, optimizer):

    model.train()
    optimizer.zero_grad()
    out, x_latent = model(dataset.x, dataset.edge_index)

    pred = out.argmax(dim=1)

    # train
    # number of correct node predictions
    train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]
    train_acc = int(train_correct.sum()) / \
        int(dataset.train_mask.sum())  # training_accuracy

    loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])

    loss.backward()
    optimizer.step()

    return loss, train_acc, x_latent


def val_epoch(dataset, model, criterion):

    model.eval()
    out, _ = model(dataset.x, dataset.edge_index)

    pred = out.argmax(dim=1)

    val_correct = pred[dataset.val_mask] == dataset.y[dataset.val_mask]
    val_acc = int(val_correct.sum()) / int(dataset.val_mask.sum())

    loss_val = criterion(out[dataset.val_mask], dataset.y[dataset.val_mask])

    return loss_val, val_acc


def train(criterion,optimizer,loader,loader_val,model,filename_save,filename_best_model,filename_visus,filename_templates,filename_alpha,filename_current_model,save,schedule, template_sizes,nepochs):

    best_val_perf = 0
    Templates = []
    alphas = []
    if schedule:
        scheduler = StepLR(optimizer, 200, 0.8)

    if save:
      save_templates = args['model'] == 'LTFGW_MLP' or  args['model'] == 'LTFGW_GCN' or  args['model'] == 'LTFGW_MLP_log' or  args['model'] == 'LTFGW_MLP_dropout' or  args['model'] == 'LTFGW_MLP_semirelaxed' or args['model'] == 'LTFGW_MLP_dropout' or  args['model'] == 'LTFGW_MLP_dropout_relu'
    
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
        df.to_pickle(filename_save)

    for epoch in tqdm(range(nepochs)):
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        start = time.time()
        for dataset in loader:
            loss, train_acc, x_latent = train_epoch(
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
            df = pd.read_pickle(filename_save)
            # add performances to the dataframe
            df.at[epoch, 'loss'] = mean_train_loss
            df.at[epoch, 'train_accuracy'] = mean_train_acc
            df.at[epoch, 'validation_accuracy'] = mean_val_acc
            df.at[epoch, 'loss_validation'] = mean_val_loss

            if  mean_val_acc > best_val_perf:

                # save best model parameters
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           filename_best_model)
                best_val_perf = mean_val_acc
                df.at[epoch, 'best_validation_accuracy'] = mean_val_acc

                # save latent embedding for visualisation
                x_latent = x_latent.detach().numpy()
                df_x = pd.DataFrame(x_latent)
                df_x.to_csv(filename_visus)

            df.to_pickle(filename_save)
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       filename_current_model)

            if save_templates and template_sizes==None:

                df = torch.load(filename_current_model)
                Templates.append(df['model_state_dict']['LTFGW.templates'])
                alphas.append(df['model_state_dict']['LTFGW.alpha0'])

        end = time.time()
        # print performances
        print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {mean_train_loss:.4f},Loss validation: {mean_val_loss:.4f},Train Accuracy: {mean_train_acc:.4f},Validation Accuracy:{mean_val_acc:.4f}')

        if save_templates:
            torch.save(Templates, filename_templates)
            torch.save(alphas, filename_alpha)

        if schedule:
            scheduler.step()


def test(model, dataset, test_graph):
    """"
    test the model
    test_graph: True if the test is done on the whole dataset, False if it is a subset of the dataset
    """
    model.eval()
    out, _ = model(dataset.x, dataset.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    if test_graph:
        test_correct = pred == dataset.y
    else:
        test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]
    test_acc = int(test_correct.sum()) / len(dataset.x[dataset.test_mask])
    return test_acc


def train_epoch_multi_graph(model, criterion, optimizer, train_loader):

    model.train()
    optimizer.zero_grad()
    Loss = []
    Train_acc = []
    X_latent = []
    Data = []
    for data in train_loader:  # Iterate in batches over the training dataset.
        Data.append(data)  # save data for visualisation

        # Perform a single forward pass.
        out, x_latent = model(data.x, data.edge_index)

        pred = out.argmax(dim=1)

        train_correct = pred == data.y  # number of correct node predictions
        train_acc = int(train_correct.sum()) / \
            int(len(data.y))  # training_accuracy

        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        Loss.append(loss)
        Train_acc.append(train_acc)
        X_latent.append(x_latent.detach().numpy())

    Loss = torch.Tensor(Loss)
    Train_acc = torch.Tensor(Train_acc)

    return torch.mean(Loss), torch.mean(Train_acc), X_latent, Data


def validation_epoch_multi_graph(model, val_loader, criterion):
    model.eval()
    Val_acc = []
    X_latent = []
    Data = []
    Loss_val = []

    for data in val_loader:  # Iterate in batches over the training dataset.
        Data.append(data)  # save data for visualisation
        # Perform a single forward pass.
        out, x_latent = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        Loss_val.append(criterion(out, data.y))
        val_correct = pred == data.y  # number of correct node predictions
        val_acc = int(val_correct.sum()) / \
            int(len(data.y))  # training_accuracy
        Val_acc.append(val_acc)
        X_latent.append(x_latent.detach().numpy())

    Loss_val = torch.Tensor(Loss_val)
    Val_acc = torch.Tensor(Val_acc)

    return torch.mean(Val_acc), X_latent, Data, torch.mean(Loss_val)


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
        loss, train_acc, x_latent, Data = train_epoch_multi_graph(
            model, criterion, optimizer, train_loader)
        end = time.time()

        # validation
        val_acc, x_latent_val, Data_validation, loss_val = validation_epoch_multi_graph(
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

                # save latent embedding for visualisation: train and validation
                # data
                df_x = pd.DataFrame(x_latent)
                df_x.to_pickle(filename_visus)
                Data = pd.DataFrame(Data)
                Data.to_pickle('Data_' + filename_visus)

                x_latent_val = x_latent_val
                df_x_val = pd.DataFrame(x_latent_val)
                df_x_val.to_pickle('validation_' + filename_visus)
                Data_validation = pd.DataFrame(Data_validation)
                Data_validation.to_pickle('Data_' + filename_visus)

            df.to_pickle(filename_save)


def test_multigraph(model, dataset):
    """"
    test the model
    """
    model.eval()
    Test_acc = []
    for data in dataset:
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred == data.y
        test_acc = int(test_correct.sum()) / len(data.x)
        Test_acc.append(test_acc)
    Test_acc = torch.tensor(Test_acc)
    return torch.mean(Test_acc)


def train_epoch_minibatch(data, criterion, optimizer, model, loader):
    model.train()
    total_loss = []
    total_train_acc = 0
    optimizer.zero_grad()
    for data in loader:
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        train_correct = pred[data.train_mask] == data.y[data.train_mask]
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
        total_train_acc += train_acc
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        total_loss.append(loss)

    mean_loss = torch.mean(torch.stack(total_loss))
    mean_loss.backward()
    optimizer.step()

    return mean_loss, total_train_acc / len(loader)


def validation_epoch_minibatch(model, loader, criterion):
    """"
    test the model
    test_graph: True if the test is done on the whole dataset, False if it is a subset of the dataset
    """
    model.eval()
    total_val_acc = 0
    total_loss_val = 0
    for data in loader:
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        val_correct = pred[data.val_mask] == data.y[data.val_mask]
        val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
        total_val_acc += val_acc
        loss_val = criterion(out[data.val_mask], data.y[data.val_mask])
        total_loss_val += loss_val

    return val_acc / len(loader), total_loss_val / len(loader)


def train_minibatch(
        model,
        dataset,
        n_epoch,
        criterion,
        optimizer,
        save,
        filename_save,
        filename_best_model,
        filename_visus,
        loader,
        loader_val,
        filename_current_model):

    best_val_perf = 0

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
        df.to_pickle(filename_save)

    train_nodes = torch.where(dataset.train_mask)
    val_nodes = torch.where(dataset.val_mask)
    nodes = torch.hstack([train_nodes[0], val_nodes[0]])
    n_nodes = len(nodes)

    for epoch in tqdm(range(n_epoch)):

        # choose random indices to train on
        indices = torch.randperm(n_nodes)[:100]

        loader = NeighborLoader(dataset,
                                num_neighbors=[-1],
                                input_nodes=indices,
                                batch_size=200)

        start = time.time()
        loss, train_acc = train_epoch_minibatch(
            dataset, criterion, optimizer, model, loader)
        val_acc, loss_val = validation_epoch_minibatch(
            model, loader, criterion)
        end = time.time()

        if save:
            df = pd.read_pickle(filename_save)
            # add performances to the dataframe
            df.at[epoch, 'loss'] = loss.item()
            df.at[epoch, 'train_accuracy'] = train_acc
            df.at[epoch, 'validation_accuracy'] = val_acc
            df.at[epoch, 'loss_validation'] = loss_val.item()

            if val_acc > best_val_perf:

                # save best model parameters
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           filename_best_model)
                best_val_perf = val_acc
                df.at[epoch, 'best_validation_accuracy'] = val_acc

            df.to_pickle(filename_save)
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       filename_current_model)

        # print performances
        print(f'Epoch: {epoch:03d},time:{end-start:.4f}, Loss: {loss:.4f},Loss validation: {loss_val:.4f},Train Accuracy: {train_acc:.4f},Validation Accuracy:{val_acc:.4f}')


def test_minibatch(model, loader):
    """"
    test the model
    test_graph: True if the test is done on the whole dataset, False if it is a subset of the dataset
    """
    model.eval()
    total_test_acc = 0
    for data in tqdm(loader):
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.train_mask.sum())
        total_test_acc += test_acc
    return test_acc / len(loader)


def test_minibatch(model, loader, criterion):
    """"
    test the model
    test_graph: True if the test is done on the whole dataset, False if it is a subset of the dataset
    """
    model.eval()
    total_test_acc = 0
    total_loss_test = 0
    for data in loader:
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        total_test_acc += test_acc
        loss_test = criterion(out[data.test_mask], data.y[data.test_mask])
        total_loss_test += loss_test

    return test_acc / len(loader), total_loss_test / len(loader)
