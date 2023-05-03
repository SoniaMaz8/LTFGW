import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear
from layers import LTFGW
import torch.nn.functional as F

class GCN_LTFGW_parallel(nn.Module):
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
        
        self.conv1=GCNConv(self.N_features, self.hidden_layer)
        self.conv2=GCNConv(self.N_features, self.hidden_layer) 
        self.LTFGW=LTFGW(self.N_templates,self.N_templates_nodes, self.hidden_layer,self.alpha0)
        self.linear=Linear(self.N_templates+self.hidden_layer, self.n_classes)
        self.batch_norm=torch.nn.BatchNorm1d(self.hidden_layer+self.N_templates)

    def forward(self, x, edge_index):
        x_LTFGW=self.conv1(x,edge_index)
        x_LTFGW= x_LTFGW.relu()
        y=self.LTFGW(x_LTFGW,edge_index)
        x=self.conv2(x,edge_index)
        x=x.relu()
        x=torch.hstack([x,y])
        x=self.batch_norm(x)
        x=self.linear(x)
        return x   
   
    

class GCN_2_layers(nn.Module):
    def __init__(self,n_classes=2,N_features=10,hidden_layer=20):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_features=N_features
        self.hidden_layer=hidden_layer
        
        self.conv1=GCNConv(self.N_features,self.hidden_layer)
        self.conv2=GCNConv(self.hidden_layer,self.n_classes)

    def forward(self, x, edge_index):
        x=self.conv1(x, edge_index)
        x=F.relu(x) 
        x=self.conv2(x, edge_index) 
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
        
        self.conv1=GCNConv(self.N_features, self.hidden_layer)
        self.LTFGW=LTFGW(self.N_templates,self.N_templates_nodes, self.hidden_layer,self.alpha0,self.q0)
        self.linear=Linear(self.N_templates+self.hidden_layer, self.n_classes)
        self.batch_norm=torch.nn.BatchNorm1d(self.hidden_layer+self.N_templates)

    def forward(self, x, edge_index):
        x=self.LTFGW(x,edge_index)
        x=x.relu()
        x=self.conv1(x,edge_index)
        return  x      
    

class MLP(nn.Module):
    def __init__(self,n_classes=2,N_features=10,hidden_layer=10):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_features=N_features
        self.hidden_layer=hidden_layer
        
        self.linear1=Linear(self.N_features, self.hidden_layer)
        self.linear2=Linear(self.hidden_layer, self.n_classes)

    def forward(self, x, edge_index):
        x=self.linear1(x,edge_index)
        x=x.relu()
        x=self.linear2(x,edge_index)
        return  x   
   
          
