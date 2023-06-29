# %%

import torch_geometric
from torch_geometric.loader import NeighborLoader, DataLoader, ClusterData, ClusterLoader, DataLoader
from GNN.architectures import *

from GNN.utils import *
from main.trainers import *
import os
import torch
import numpy as np
import torch_geometric.transforms as T
import argparse

torch_geometric.typing.WITH_PYG_LIB = False

# %%Parameters to set




parser = argparse.ArgumentParser(description='Graph node classification')

parser.add_argument('-dataset', type=str, default='cornell_directed',
                    help='name of the dataset to use')
parser.add_argument('-model', type=str, default='MLP',
                    help='name of the model to use')
parser.add_argument('-nepochs', type=int, default=1000,
                    help='number of epochs')
parser.add_argument(
    '-graph_type',
    type=str,
    default='single_graph',
    help='type of graph in : single_graph, multi_graph, mini_batch')
parser.add_argument('-lr', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('-wd', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('-n_templates', type=int, default=1,
                    help='number of templates for LTFGW')
parser.add_argument('-n_templates_nodes', type=int, default=2,
                    help='number of templates nodes for LTFGW')
parser.add_argument('-hidden_layer', type=int, default=64,
                    help='hidden dimention')
parser.add_argument('-n_hidden_layer', type=int, default=0,
                    help='number of hidden layers')
parser.add_argument('-batch_size', type=int, default=5,
                    help='batch size if multi graph')
parser.add_argument('-first_seed', type=int, default=20,
                    help='first seed to train with')
parser.add_argument(
    '-number_of_seeds',
    type=int,
    default=1,
    help='number of seeds to train with, starting from the first')
parser.add_argument('-train_node_weights', type=str, default='True',
                    help='wether to train the template node weights')
parser.add_argument('-save', type=str, default='True',
                    help='wether to save the results')
parser.add_argument('-random_split', type=list, default=[600, 200, 200],
                    help='size of train/val/test for multigraph')
parser.add_argument('-alpha0', type=float, default=None,
                    help='alpha0 for LTFGW')
parser.add_argument(
    '-k',
    type=int,
    default=1,
    help='nomber of hops (order of the neighbourhood) in LTFGW')
parser.add_argument('-dropout', type=float, default=0.6,
                    help='dropout')
parser.add_argument('-shortest_path', type=str, default='False',
                    help='wether to use the shortest path cost matrix in TFGW')
parser.add_argument('-skip_connection', type=str, default='True',
                    help='wether to skip connection is the architectures')
parser.add_argument('-scheduler', type=str, default='False',
                    help='wether to use a learning rate scheduler')
parser.add_argument('-template_sizes', type=int,nargs='+', default=None,
                    help='list of template sizes')
parser.add_argument('-reg', type=float, default=0,
                    help='regularisation for entropic semi_relaxed')
parser.add_argument('-log', type=str, default='False',
                    help='If True, the log of the LTFGW is used')
# parser.add_argument('-seeds', type=list, default=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363],
#                    help='seeds to use for splits')


args = vars(parser.parse_args())



args['save'] = args['save'] == 'True'  
args['scheduler'] = args['scheduler'] == 'True'
args['train_node_weights'] = args['train_node_weights'] == 'True'
args['shortest_path'] = args['shortest_path'] == 'True'

if not args['template_sizes'] == None:
   args['n_templates']=len(args['template_sizes'])


if not args['alpha0'] is None:
    args['alpha0'] = torch.as_tensor([args['alpha0']])

args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%Training and testing

criterion = torch.nn.CrossEntropyLoss()

Test_accuracy = []
seeds = torch.arange(args['first_seed'], args['first_seed'] + args['num_seeds'], 1)

# load dataset
dataset, args['n_classes'], args['n_features'], test_graph, graph_type, args['mean_init'], args['std_init'] = get_dataset(
    args['dataset_name'])

args['n_nodes'] = len(dataset.x)

if args['dataset_name'] == 'Toy_graph_single':
    dataset_train, dataset_test = dataset
    dataset = dataset_train
else:
    dataset_test = dataset

method = args['model_name']    

for seed in seeds:
    torch.manual_seed(seed)

    model = get_model(**args)
    model = model.to(args['device'])
     
    if args['alpha0'] is None:
        filename_save, filename_best_model, filename_visus, filename_current_model, filename_templates, filename_alpha = get_filenames(seed=seed,method=method,**args)
    else:
        filename_save, filename_best_model, filename_visus, filename_current_model, filename_templates, filename_alpha = get_filenames(seed=seed,method=method,**args)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay'])

    if graph_type == 'single_graph':
        percls_trn = int(round(0.6 * len(dataset.y) / args['n_classes']))
        val_lb = int(round(0.2 * len(dataset.y)))
        dataset = random_planetoid_splits(
            dataset,
            args['n_classes'],
            percls_trn=percls_trn,
            val_lb=val_lb,
            seed=seed)
        dataset = dataset.to(args['device'])
        loader = NeighborLoader(dataset,
                                num_neighbors=[-1],
                                input_nodes=dataset.train_mask,
                                directed=False,
                                batch_size=torch.sum(dataset.train_mask).item()
                                )
        loader_val = NeighborLoader(dataset,
                                    num_neighbors=[-1],
                                    input_nodes=dataset.val_mask,
                                    directed=False,
                                    batch_size=torch.sum(dataset.val_mask).item()
                                    )
        dataset_test = dataset
        train(**args,criterion,optimizer,loader,loader_val,model,filename_save,filename_best_model,filename_visus,filename_templates,filename_alpha,filename_current_model)

    elif graph_type == 'multi_graph':
        generator = torch.Generator().manual_seed(seed.item())
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, args['random_split'], generator=generator)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            shuffle=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args['batch_size'],
            shuffle=False)
        train_multi_graph(
            model,
            criterion,
            optimizer,
            train_loader,
            val_loader)

    elif graph_type == 'mini_batch':
        percls_trn = int(round(0.6 * len(dataset.y) / args['n_classes']))
        val_lb = int(round(0.2 * len(dataset.y)))
        dataset = random_planetoid_splits(
            dataset,
            args['n_classes'],
            percls_trn=percls_trn,
            val_lb=val_lb,
            seed=seed)

        train_minibatch(
            model,
            dataset,
            args['n_epoch'],
            criterion,
            optimizer,
            args['save'] ,
            filename_save,
            filename_best_model,
            filename_visus,
            loader,
            loader_val,
            filename_current_model)

    checkpoint = torch.load(filename_best_model)
    model.load_state_dict(checkpoint['model_state_dict'])

    if graph_type == 'multi_graph':
        test_acc = test_multigraph(model, dataset_test)
        Test_accuracy.append(test_acc)

    elif graph_type == 'single_graph':
        test_acc = test(model, dataset_test, test_graph)
        Test_accuracy.append(test_acc)

    elif graph_type == 'mini_batch':
        test_acc = test(model, dataset_test, test_graph)
        Test_accuracy.append(test_acc)

    if args['template_sizes']==None:

        filename_save_test = os.path.join(
            'results',
            method,
            args['dataset_name'],
            "test_seed{}_lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler{}_log{}.csv".format(args['seed'],args['lr'],args['n_templates'],args['n_templates_nodes'],
                args['alpha0'],args['k'],args['dropout'],args['wd'],args['hidden_layer'],args['scheduler'],args['log']))
    else: 

        filename_save_test = os.path.join(
            'results',
            method,
            args['dataset_name'],
            "test_seed{}_lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler{}_log{}_tempsizes.csv".format(args['seed'],args['lr'],args['n_templates'],args['n_templates_nodes'],
                args['alpha0'],args['k'],args['dropout'],args['wd'],args['hidden_layer'],args['scheduler'],args['log']))
           
       
    if not os.path.isdir(os.path.join('results',method,args['dataset_name'])):
        os.mkdir(os.path.join('results',method,args['dataset_name']))
    np.savetxt(filename_save_test, Test_accuracy)

# print the performances
Test_accuracy = torch.tensor(Test_accuracy)
print(
    'mean_accuracy={},std={}'.format(
        torch.mean(Test_accuracy),
        torch.std(Test_accuracy)))


# %%
