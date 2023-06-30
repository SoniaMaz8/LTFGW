# %%

import torch_geometric
from torch_geometric.loader import NeighborLoader, DataLoader, ClusterData, ClusterLoader, DataLoader
from GNN.architectures import *

from main.utils import *
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
parser.add_argument('-n_template_nodes', type=int, default=2,
                    help='number of templates nodes for LTFGW')
parser.add_argument('-hidden_layer', type=int, default=2,
                    help='hidden dimention')
parser.add_argument('-n_hidden_layer', type=int, default=0,
                    help='number of hidden layers')
parser.add_argument('-batch_size', type=int, default=5,
                    help='batch size if multi graph')
parser.add_argument('-first_seed', type=int, default=20,
                    help='first seed to train with')
parser.add_argument(
    '-n_seeds',
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



save = args['save'] == 'True'  
scheduler = args['scheduler'] == 'True'
train_node_weights = args['train_node_weights'] == 'True'
shortest_path = args['shortest_path'] == 'True'
log=args['log']=='True'
skip_connection=args['skip_connection']=='True'

if not args['template_sizes'] == None:
   args['n_templates']=len(args['template_sizes'])

alpha0=args['alpha0']
if not alpha0 is None:
    alpha0 = torch.as_tensor([args['alpha0']])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_templates=args['n_templates']
n_template_nodes=args['n_template_nodes']
dataset_name=args['dataset']
hidden_layer=args['hidden_layer']
dropout=args['dropout']
wd=args['wd']
lr=args['lr']
k=args['k']
template_sizes=args['template_sizes']
n_hidden_layer=args['n_hidden_layer']
reg=args['reg']
nepochs=args['nepochs']
model_name=args['model']



# %%Training and testing

criterion = torch.nn.CrossEntropyLoss()

Test_accuracy = []
seeds = torch.arange(args['first_seed'], args['first_seed'] + args['n_seeds'], 1)

# load dataset
dataset, n_classes, n_features, test_graph, graph_type, mean_init, std_init = get_dataset(
    args['dataset'])

n_nodes = len(dataset.x)

if dataset_name == 'Toy_graph_single':
    dataset_train, dataset_test = dataset
    dataset = dataset_train
else:
    dataset_test = dataset

method = args['model']    

for seed in seeds:
    torch.manual_seed(seed)

    model = get_model( model_name,
        n_classes,
        n_features,
        n_templates,
        n_template_nodes,
        hidden_layer,
        dropout,
        shortest_path,
        k,
        mean_init,
        std_init,
        log,
        alpha0,
        train_node_weights,
        skip_connection,
        template_sizes,
        n_hidden_layer,
        reg)
    model = model.to(device)
     
    if alpha0 is None:
        filename_save, filename_best_model, filename_visus, filename_current_model, filename_templates, filename_alpha = get_filenames(args['dataset'],
        method,
        lr,
        n_templates,
        n_template_nodes,
        alpha0,
        k,
        dropout,
        wd,
        hidden_layer,
        scheduler,
        seed,
        template_sizes,
        log)
    else:
        filename_save, filename_best_model, filename_visus, filename_current_model, filename_templates, filename_alpha = get_filenames(args['dataset'],
        method,
        lr,
        n_templates,
        n_nodes,
        alpha0.item(),
        k,
        dropout,
        wd,
        hidden_layer,
        scheduler,
        seed,
        template_sizes,
        log)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd)

    if graph_type == 'single_graph':
        percls_trn = int(round(0.6 * len(dataset.y) / n_classes))
        val_lb = int(round(0.2 * len(dataset.y)))
        dataset = random_planetoid_splits(
            dataset,
            n_classes,
            percls_trn=percls_trn,
            val_lb=val_lb,
            seed=seed)
        dataset = dataset.to(device)
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
        train(criterion,optimizer,loader,loader_val,model,filename_save,filename_best_model,filename_visus,filename_templates,filename_alpha,filename_current_model,save,scheduler, template_sizes,nepochs,model_name)

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
        percls_trn = int(round(0.6 * len(dataset.y) / n_classes))
        val_lb = int(round(0.2 * len(dataset.y)))
        dataset = random_planetoid_splits(
            dataset,
            n_classes,
            percls_trn=percls_trn,
            val_lb=val_lb,
            seed=seed)

        train_minibatch(
            model,
            dataset,
            nepochs,
            criterion,
            optimizer,
            save ,
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
            args['dataset'],
            "test_seed{}_lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler{}_log{}.csv".format(seed,lr,n_templates,n_template_nodes,
                alpha0,k,dropout,wd,hidden_layer,scheduler,log))
    else: 

        filename_save_test = os.path.join(
            'results',
            method,
            args['dataset'],
            "test_seed{}_lr{}_n_temp{}_n_nodes{}_alpha0{}_k{}_drop{}_wd{}_hl{}_scheduler{}_log{}_tempsizes.csv".format(seed,lr,n_templates,n_template_nodes,
                alpha0,k,dropout,wd,hidden_layer,scheduler,log))
           
       
    if not os.path.isdir(os.path.join('results',method,dataset_name)):
        os.mkdir(os.path.join('results',method,dataset_name))
    np.savetxt(filename_save_test, Test_accuracy)

# print the performances
Test_accuracy = torch.tensor(Test_accuracy)
print(
    'mean_accuracy={},std={}'.format(
        torch.mean(Test_accuracy),
        torch.std(Test_accuracy)))


# %%
