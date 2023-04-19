import torch
import torch.nn as nn
from utils import distance_to_template
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv, Linear


class LTFWG(nn.Module):
    """ Layer for the local TFWG """
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

        latent_template=torch.Tensor(self.N_templates,self.N_templates_nodes,self.N_templates_nodes)
        self.latent_template=nn.Parameter(latent_template)

        templates_features=torch.Tensor(self.N_templates,self.N_templates_nodes,self.N_features)
        self.templates_features =nn.Parameter(templates_features)

        # initialize adjacency matrices for the templates
        nn.init.uniform_(self.latent_template)
        nn.init.uniform_(self.templates_features)

    def forward(self, x, edge_index):
        template=0.5*(self.latent_template+torch.transpose(self.latent_template,1,2))
        x=distance_to_template(x,edge_index,self.templates_features,template)
        return x

class OT_GNN_layer(nn.Module):
    def __init__(self,n_classes=2,N_features=10, N_templates=10,N_templates_nodes=10):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.num_classes=n_classes

        self.LTFWG=LTFWG(N_features, N_templates,N_templates_nodes)
        self.linear=Linear(N_templates, self.num_classes)

    def forward(self, x, edge_index):
        x=self.LTFWG(x,edge_index)
        x=self.linear(x)
        return  x
