import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear
from layers import LTFGW
import torch.nn.functional as F

class GCN_LTFGW(nn.Module):
    def __init__(self,n_classes=2,N_features=10, N_templates=10,N_templates_nodes=10,hidden_layer=20,alpha0=None, skip_connection=False):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_features=N_features
        self.N_templates=N_templates
        self.N_templates_nodes=N_templates_nodes
        self.hidden_layer=hidden_layer
        self.alpha0=alpha0
        self.skip_connection=skip_connection
        
        self.conv1=GCNConv(self.N_features, self.hidden_layer)
        self.conv2=GCNConv(self.N_features, self.hidden_layer) 
        self.LTFGW=LTFGW(self.N_templates,self.N_templates_nodes, self.hidden_layer,self.alpha0)
        self.linear=Linear(self.N_templates+self.hidden_layer, self.n_classes)
        self.batch_norm=torch.nn.BatchNorm1d(self.hidden_layer+self.N_templates)

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
        
        # final prediction
        x=self.linear(x)
        return x   
   
    

class GCN(nn.Module):
    def __init__(self,n_classes=2,N_features=10,hidden_layer=20, n_hidden_layers=0):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_features=N_features
        self.hidden_layer=hidden_layer
        self.n_hidden_layers=n_hidden_layers
        
        self.first_conv=GCNConv(self.N_features,self.hidden_layer)

        # list of GCN layers
        self.list_hidden_layer = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.list_hidden_layer.append(GCNConv(self.hidden_layer,self.hidden_layer))
        
        self.last_conv=GCNConv(self.hidden_layer,self.n_classes)

    def forward(self, x, edge_index):

        x=self.first_conv(x,edge_index)
        x=x.relu()

        # go through hidden layers

        for i in range(self.n_hidden_layers):
            x=self.list_hidden_layer[i](x,edge_index)
            x=x.relu()

        x=self.last_conv(x, edge_index) 
        return  x   
    


class LTFGW_GCN(nn.Module):
    def __init__(self,n_classes=2,N_features=10, N_templates=10,N_templates_nodes=10,hidden_layer=20,alpha0=None,q0=None):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_features=N_features
        self.N_templates=N_templates
        self.N_templates_nodes=N_templates_nodes
        self.hidden_layer=hidden_layer
        self.alpha0=alpha0
        self.q0=q0
        
        self.conv1=GCNConv(self.N_templates, self.n_classes)
        self.LTFGW=LTFGW(self.N_templates,self.N_templates_nodes, self.N_features,self.alpha0,self.q0)

    def forward(self, x, edge_index):
        x=self.LTFGW(x,edge_index)
        x=x.relu()
        x=self.conv1(x,edge_index)
        return  x      
    

class MLP(nn.Module):
    def __init__(self,n_classes=2,n_features=10,hidden_layer=20, n_hidden_layers=1):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.n_features=n_features
        self.hidden_layer=hidden_layer
        self.n_hidden_layers=n_hidden_layers

        # list of Linear layers
        self.hidden_layer = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.hidden_layer.append(Linear(self.N_features,self.hidden_layer))

        self.last_linear=Linear(self.hidden_layer, self.n_classes)

    def forward(self, x, edge_index):
        # go through hidden layers
        for i in range(self.n_hidden_layers):
            x=self.hidden_layer[i](x,edge_index)
            x=x.relu()
            
        x=self.linear2(x,edge_index)
        return  x   
   
          
