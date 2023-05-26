import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear
from GNN.layers import LTFGW
import torch.nn.functional as F
from sklearn.manifold import TSNE
import math 
from torch.nn.parameter import Parameter


class GCN_LTFGW(nn.Module):
    def __init__(self,n_classes=2,n_features=10, n_templates=10,n_templates_nodes=10,hidden_layer=20,alpha0=None, skip_connection=False):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.n_features=n_features
        self.n_templates=n_templates
        self.n_templates_nodes=n_templates_nodes
        self.hidden_layer=hidden_layer
        self.alpha0=alpha0
        self.skip_connection=skip_connection
        
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
    def __init__(self,n_classes=2,n_features=10,hidden_layer=10, n_hidden_layers=0):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.n_features=n_features
        self.hidden_layer=hidden_layer
        self.n_hidden_layers=n_hidden_layers

        self.first_conv=GCNConv(self.n_features,self.hidden_layer)
        self.dropout=torch.nn.Dropout(p=0.5)

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
    def __init__(self,n_classes=2,n_features=10, n_templates=10,n_templates_nodes=10,hidden_layer=10,alpha0=None,train_node_weights=True, skip_connection=True):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.n_features=n_features
        self.n_templates=n_templates
        self.n_templates_nodes=n_templates_nodes
        self.hidden_layer=hidden_layer
        self.alpha0=alpha0
        self.train_node_weights=train_node_weights
        self.skip_connection=skip_connection
        
        self.conv1=GCNConv(self.n_features, self.hidden_layer)
        self.conv2=GCNConv(self.hidden_layer+self.n_templates, self.n_classes)
        self.conv3=GCNConv(self.n_templates, self.n_classes)
        self.LTFGW=LTFGW(self.n_templates,self.n_templates_nodes, self.n_features,self.alpha0,self.train_node_weights)
        self.batch_norm=torch.nn.BatchNorm1d(self.hidden_layer+self.n_templates)
        

    def forward(self, x, edge_index):

        if self.skip_connection:
            y=self.LTFGW(x,edge_index)
            z=self.conv1(x,edge_index)
            z=z.relu()
            x = torch.hstack([z,y])
            x=self.batch_norm(x)
            x_latent=x
            x=self.conv2(x,edge_index)
        else:
            x=self.LTFGW(x,edge_index)
            x=self.conv3(x,edge_index)
        return  x,x_latent
    

class MLP(nn.Module):
    def __init__(self,n_classes=2,n_features=10,hidden_layer=20, n_hidden_layers=0):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()

    
        self.n_classes=n_classes
        self.n_features=n_features
        self.hidden_layer=hidden_layer
        self.n_hidden_layers=n_hidden_layers

        self.first_linear=Linear(self.n_features, self.hidden_layer)
        self.dropout1=torch.nn.Dropout(0.5)
        self.dropout2=torch.nn.Dropout(0.5)
        
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
    def __init__(self,n_nodes,n_classes=2,n_features=10, n_templates=10,n_templates_nodes=10,hidden_layer=10,alpha0=None,train_node_weights=True, skip_connection=True):
        """
        n_classes: number of classes for node classification
        n_features: number of features for each node
        n_templates: number of templates to use for LTFGW
        n_templates_nodes: number of nodes for each template for LTFGW
        hidden_layer: number of hidden dimensions
        alpha0: alpha paramameter for Fused Gromov Wasserstein, if None it is learned
        train_node_weights: wether to learn node weights on the templates for LFTGW
        skip_connection: wether to put MLP and LTFGW in parallel

        """
        super().__init__()
    
        self.n_classes=n_classes
        self.n_features=n_features
        self.n_templates=n_templates
        self.n_templates_nodes=n_templates_nodes
        self.hidden_layer=hidden_layer
        self.alpha0=alpha0
        self.train_node_weights=train_node_weights
        self.skip_connection=skip_connection
        self.n_nodes=n_nodes

        self.dropout2=torch.nn.Dropout(0.5)
        
        self.Linear1=Linear(self.n_features, self.hidden_layer)
        self.Linear2=Linear(self.hidden_layer+self.n_templates, self.n_classes)
        self.Linear3=Linear(self.n_templates, self.n_classes)
        self.LTFGW=LTFGW(self.n_nodes,self.n_templates,self.n_templates_nodes, self.hidden_layer,self.alpha0,self.train_node_weights)
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
   
