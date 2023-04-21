import torch
import torch.nn as nn
from utils import distance_to_template
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv, Linear
import torch.nn.functional as F


class LTFGW(nn.Module):
    """ Layer for the local TFGW """
    def __init__(self, N_features=10, N_templates=10,N_templates_nodes=10):
        """
        N_features: number of node features
        N_templates: number of graph templates
        N_templates_nodes: number of nodes in each template
        """
        super().__init__()
        self.N_templates=N_templates
        self.N_templates_nodes=N_templates_nodes
        self.N_features=N_features

        templates_features=torch.Tensor(self.N_templates,self.N_templates_nodes,self.N_features)
        nn.init.normal_(templates_features)
        self.templates_features = nn.Parameter(F.normalize(templates_features, p=2, dim=2))
        
        template=torch.Tensor(self.N_templates,self.N_templates_nodes,self.N_templates_nodes)
        nn.init.normal_(template)
        template=F.normalize(template, p=2, dim=2)
        template=0.5*(template+torch.transpose(template,1,2))
        self.template=template
        
    def forward(self, x, edge_index):
        x=distance_to_template(x,edge_index,self.templates_features,template)
        return x

class OT_GNN_layer(nn.Module):
    def __init__(self,n_classes=2,N_features=10, N_templates=10,N_templates_nodes=10):
        """
        n_classes: number of classes for node classification
        N_features: number of node features
        N_templates: number of templates
        N_templates_nodes: number of nodes in each template
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_features=N_features
        self.N_templates=N_templates
        self.N_templates_nodes=N_templates_nodes

        self.LTFWG=LTFGW(self.N_features, self.N_templates,self.N_templates_nodes)
        self.linear=Linear(self.N_templates, self.n_classes)

    def forward(self, x, edge_index):
        x=self.LTFGW(x,edge_index)
        x=self.linear(x)
        return  x
