import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear
from layers import LTFGW
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN_LTFGW(nn.Module):
    def __init__(self,n_classes=2,N_features=10, N_templates=10,N_templates_nodes=10,hidden_layer=20,dropout=0.5,alpha0=None):
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
        self.dropout=dropout
        
        self.conv1=GCNConv(self.N_features,self.hidden_layer)
        self.conv2=GCNConv(self.hidden_layer,self.hidden_layer) 
        self.conv3=GCNConv(self.N_templates+self.hidden_layer,self.n_classes)  
        self.LTFGW=LTFGW(self.N_templates,self.N_templates_nodes,self.hidden_layer,self.alpha0)
        self.linear=Linear(self.N_templates+self.hidden_layer, self.n_classes)

    def forward(self, x, edge_index):
        x=self.conv1(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu() 
        x=self.conv2(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        y=self.LTFGW(x,edge_index)
        x=torch.hstack([x,y])
        x=self.conv3(x, edge_index) 
        return  x    
    

class GCN_3_layers(nn.Module):
    def __init__(self,n_classes=2,N_features=10,hidden_layer=16,dropout=0.5):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_features=N_features
        self.hidden_layer=hidden_layer
        self.dropout=dropout
        
        self.conv1=GCNConv(self.N_features,self.hidden_layer)
        self.conv2=GCNConv(self.hidden_layer,self.hidden_layer)
        self.conv3=GCNConv(self.hidden_layer,self.n_classes)

    def forward(self, x, edge_index):
        x=self.conv1(x, edge_index)
        x=F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
    #    x=self.conv2(x, edge_index)
    #    x=F.relu(x)
    #    x = F.dropout(x, self.dropout, training=self.training)       
        x=self.conv3(x, edge_index) 
        return  x   
    
   
          
