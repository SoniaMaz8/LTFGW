import torch
import torch.nn as nn
from utils import distance_to_template
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv, Linear


class LTFWG(nn.Module):
    """ Layer for the local TFWG """
    def __init__(self, N_features=3, N_templates=10,N_templates_nodes=5):
        """
        N_features: number of node features
        N_templates: number of graph templates
        N_templates_nodes: number of nodes in each template
        """
        super().__init__()
        self.N_templates=N_templates
        self.N_templates_nodes=N_templates_nodes
        self.N_features=N_features

        templates=torch.Tensor(self.N_templates,self.N_templates_nodes,self.N_templates_nodes)  #templates adjacency matrices 
        self.templates = nn.Parameter(templates)

        templates_features=torch.Tensor(self.N_templates,self.N_templates_nodes,self.N_features)
        self.templates_features = nn.Parameter(templates_features)

        # initialize adjacency matrices for the templates
        nn.init.uniform_(self.templates)
        nn.init.uniform_(self.templates_features)

    def forward(self, x, edge_index):
        x=distance_to_template(x,edge_index,self.templates_features,self.templates)
        return x

class OT_GNN_layer(nn.Module):
    def __init__(self,n_classes=3):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.num_classes=n_classes
        self.linear=Linear(10, self.num_classes)
        self.LTFWG=LTFWG(3,10,5)

    def forward(self, x, edge_index):
        x=self.LTFWG(x,edge_index)
        x=self.linear(x)
        return  x
    

