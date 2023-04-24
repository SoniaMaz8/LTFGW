import torch
import torch.nn as nn
from utils import distance_to_template,construct_templates,construct_templates2
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv, Linear
import torch.nn.functional as F

dataset=torch.load('data/graph_Citeseer.pt')

torch.manual_seed(123456)

class LTFGW(nn.Module):
    """ Layer for the local TFGW """
    def __init__(self, N_templates=10,N_templates_nodes=10):
        """
        N_templates: number of graph templates
        N_templates_nodes: number of nodes in each template
        """
        super().__init__()

        self.N_templates= N_templates
        self.N_templates_nodes=N_templates_nodes
        
        #templates initilisation as subgraphs of the dataset after one GCN layer

        templates,templates_features=construct_templates(dataset,self.N_templates_nodes,self.N_templates)
        templates=templates.type(torch.FloatTensor)
        self.templates=nn.Parameter(templates)
        self.templates_features = nn.Parameter(templates_features)

    def forward(self, x, edge_index):
        x=distance_to_template(x,edge_index,self.templates_features,self.templates)
        return x

class OT_GNN_layer(nn.Module):
    def __init__(self,n_classes=2, N_templates=10,N_templates_nodes=10,hidden_layer=20):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_templates=N_templates
        self.N_templates_nodes=N_templates_nodes
        self.hidden_layer=hidden_layer
        
        self.conv1=GCNConv(self.N_features,self.hidden_layer)
        self.conv2=GCNConv(self.hidden_layer,self.hidden_layer)   
        self.LTFGW=LTFGW(self.N_templates,self.N_templates_nodes)
        self.linear=Linear(self.N_templates+self.hidden_layer, self.n_classes)

    def forward(self, x, edge_index):
        x=self.conv1(x, edge_index)
        x = x.relu() 
        x=self.conv2(x, edge_index)
        y=self.LTFGW(x,edge_index)
        x=torch.hstack([x,y])
        x=self.linear(x)
        x = x.relu()  
        return  x

#LTFWG replaced by a GCN, for comparison 

class GCN_layer(nn.Module):
    def __init__(self,n_classes=2,N_features=10,hidden_layer=20):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_features=N_features
        self.hidden_layer=hidden_layer
        
        self.conv1=GCNConv(self.N_features,self.hidden_layer)
        self.conv2=GCNConv(self.hidden_layer,self.hidden_layer)
        self.conv3=GCNConv(self.hidden_layer,self.n_classes)  

    def forward(self, x, edge_index):
        x=self.conv1(x, edge_index)
        x = x.relu() 
        x=self.conv2(x, edge_index)
        x = x.relu() 
        x=self.conv3(x, edge_index)
        x = x.relu()  
        return  x        
