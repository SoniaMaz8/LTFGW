import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear, ChebConv, GATConv
from GNN.layers import LTFGW
import torch.nn.functional as F
from sklearn.manifold import TSNE
import math 
from torch.nn.parameter import Parameter


class GCN_LTFGW(nn.Module):
    def __init__(self,args,n_classes,n_features,n_nodes):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.n_features=n_features
        self.n_templates=args['n_templates']
        self.n_templates_nodes=args['n_template_nodes']
        self.hidden_layer=args['hidden_layer']
        self.alpha0=args['alpha_0']
        self.skip_connection=args['skip_connection']=='True'
        
        self.conv1=GCNConv(self.n_features, self.hidden_layer)
        self.conv2=GCNConv(self.n_features, self.hidden_layer) 
        self.LTFGW=LTFGW(self.n_templates,self.n_templates_nodes, self.hidden_layer,self.alpha0)
        self.linear=Linear(self.n_templates+self.hidden_layer, self.n_classes)
        self.batch_norm=torch.nn.BatchNorm1d(self.hidden_layer+self.n_templates)

    def forward(self, x, edge_index):

        # first conv -> dim reduction
        x=self.conv1(x,edge_index)
        x= x.relu()

        # LTFGW + batch norm
        if self.skip_connection:
            y=self.LTFGW(x,edge_index)
            x = torch.hstack([x,y])
        else:
            x=self.LTFGW(x,edge_index)
        x=self.batch_norm(x)

        # second conv -> dim reduction
        x=self.conv2(x,edge_index)
        x=x.relu()
        
        x_latent=x
        # final prediction
        x=self.linear(x)
        return x  , x_latent
   
    

class GCN(nn.Module):
    def __init__(self,args,n_classes,n_features):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.n_features=n_features
        self.hidden_layer=args['hidden_layer']
        self.n_hidden_layers=args['n_hidden_layer']
        self.dropout=args['dropout']

        self.first_conv=GCNConv(self.n_features,self.hidden_layer)
        self.dropout=torch.nn.Dropout(p=self.dropout)

        # list of GCN layers
        self.list_hidden_layer = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.list_hidden_layer.append(GCNConv(self.hidden_layer,self.hidden_layer))
        
        self.last_conv=GCNConv(self.hidden_layer,self.n_classes)

    def forward(self, x, edge_index):

        x=self.first_conv(x,edge_index)
        x=x.relu()
        x=self.dropout(x)

        # go through hidden layers

        for i in range(self.n_hidden_layers):
            x=self.list_hidden_layer[i](x,edge_index)
            x=x.relu()

        x_latent=x
        x=self.last_conv(x, edge_index) 
        return  x ,x_latent
    


class LTFGW_GCN(nn.Module):
    def __init__(self,args,n_classes,n_features,n_nodes):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.n_features=n_features
        self.n_templates=args['n_templates']
        self.n_templates_nodes=args['n_templates_nodes']
        self.hidden_layer=args['hidden_layer']
        self.alpha0=args['alpha0']
        self.train_node_weights=args['train_node_weights']=='True'
        self.skip_connection=args['skip_connection']=='True'
        self.drop=args['dropout']
        self.shortest_path=args['shortest_path']==True
        self.local_alpha=args['local_alpha']==True
        self.k=args['k']
        self.n_nodes=n_nodes

        self.dropout=torch.nn.Dropout(self.drop)
        
        self.linear=Linear(self.n_features, self.hidden_layer)
        self.conv1=GCNConv(self.n_features, self.hidden_layer)
        self.conv2=GCNConv(self.hidden_layer+self.n_templates, self.n_classes)
        self.conv3=GCNConv(self.n_templates, self.n_classes)
        self.LTFGW=LTFGW(self.n_nodes,self.n_templates,self.n_templates_nodes, self.hidden_layer,self.k,self.alpha0,self.train_node_weights,self.local_alpha,self.shortest_path)
        self.batch_norm=torch.nn.BatchNorm1d(self.hidden_layer+self.n_templates)
        

    def forward(self, x, edge_index):

        if self.skip_connection:
            y=self.linear(x)
            y=self.LTFGW(y,edge_index)
            z=self.conv1(x,edge_index)
            z=z.relu()
            x = torch.hstack([z,y])
            x=self.batch_norm(x)
            x=self.dropout(x)
            x_latent=x
            x=self.conv2(x,edge_index)
        else:
            x=self.LTFGW(x,edge_index)
            x=self.conv3(x,edge_index)
        return  x,x_latent
    

class MLP(nn.Module):
    def __init__(self,args,n_classes,n_features):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()

        self.n_classes=n_classes
        self.n_features=n_features
        self.n_hidden_layers=args['n_hidden_layer']
        self.hidden_layer=args['hidden_layer']
        self.drop=args['dropout']

        self.first_linear=Linear(self.n_features, self.hidden_layer)
        self.dropout1=torch.nn.Dropout(self.drop)
        self.dropout2=torch.nn.Dropout(self.drop)
        
        # list of Linear layers
        self.list_hidden_layer = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.list_hidden_layer.append(Linear(self.hidden_layer,self.hidden_layer))

        self.last_linear=Linear(self.hidden_layer, self.n_classes)

    def forward(self, x, edge_index):

        x=self.dropout1(x)
        x=self.first_linear(x)

        x=x.relu()

        # go through hidden layers
        for i in range(self.n_hidden_layers):
            x=self.list_hidden_layer[i](x)
            x=x.relu()

        x_latent=x 
        x=self.dropout2(x)
        x=self.last_linear(x)

        return  x  , x_latent

    

class LTFGW_MLP(nn.Module):
    def __init__(self,args,n_classes,n_features,n_nodes):
        """
        n_classes: number of classes for node classification
        n_features: number of features for each node
        n_templates: number of templates to use for LTFGW
        n_templates_nodes: number of nodes for each template for LTFGW
        hidden_layer: number of hidden dimensions
        alpha0: alpha paramameter for Fused Gromov Wasserstein, if None it is learned
        train_node_weights: wether to learn node weights on the templates for LFTGW
        skip_connection: wether to put MLP and LTFGW in parallel
        local alpha: wether to learn one tradeoff parameter for the FGW for each node or for the whole graph 

        """

        super().__init__()

        self.n_classes=n_classes
        self.n_features=n_features
        self.n_templates=args['n_templates']
        self.n_templates_nodes=args['n_templates_nodes']
        self.hidden_layer=args['hidden_layer']
        self.alpha0=args['alpha0']
        self.train_node_weights=args['train_node_weights']=='True'
        self.skip_connection=args['skip_connection']=='True'
        self.drop=args['dropout']
        self.shortest_path=args['shortest_path']==True
        self.local_alpha=args['local_alpha']==True
        self.k=args['k']
        self.n_nodes=n_nodes  

        self.dropout2=torch.nn.Dropout(self.drop)
        
        self.Linear1=Linear(self.n_features, self.hidden_layer)
        self.Linear2=Linear(self.hidden_layer+self.n_templates, self.n_classes)
        self.Linear3=Linear(self.n_templates, self.n_classes)
        self.LTFGW=LTFGW(self.n_nodes,self.n_templates,self.n_templates_nodes, self.hidden_layer,self.k,self.alpha0,self.train_node_weights,self.local_alpha,self.shortest_path)
        self.batch_norm=torch.nn.BatchNorm1d(self.hidden_layer+self.n_templates)
        

    def forward(self, x, edge_index):

        x=self.Linear1(x)
        
        if self.skip_connection:
            y=self.LTFGW(x,edge_index)
            x = torch.hstack([x,y])
            x=self.batch_norm(x)
            x=x.relu()
            x=self.dropout2(x)
            x=self.Linear2(x)
            x_latent=x

        else:
            x=self.LTFGW(x,edge_index)
            x=self.Linear3(x)

        return  x,x_latent
   
    


class ChebNet(torch.nn.Module):
    def __init__(self, args,n_classes,n_features):
        super(ChebNet, self).__init__()

        self.n_features=n_features
        self.n_classes=n_classes
        self.drop=args['dropout']

        self.conv1 = ChebConv(self.n_features, 32, K=2)
        self.conv2 = ChebConv(32, self.n_classes, K=2)
        self.dropout = self.drop

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x=self.conv1(x, edge_index)
        x = F.relu(x)
        x_latent=x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1),  x_latent
    

class GAT(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)