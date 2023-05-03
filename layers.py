import torch
import torch.nn as nn
from utils import distance_to_template
from torch_geometric.data import Data as GraphData
import torch.nn.functional as F
import pandas as pd

def template_initialisation(N_templates,N_nodes,N_features):
    """"
    Function that initialises templates for the LTFGW layer
    Input:
      N_templates: number of templates
      N_nodes: number of nodes per template
      N_features: number of features for the nodes
    Output:
      Templates: list of the adjancecy matrices of the templates
      Templates_features: list of the features of the nodes of each template
    """
    Templates=torch.randint(0, 2, size=(N_templates,N_nodes, N_nodes))
    Templates_features=torch.Tensor(N_templates,N_nodes,N_features)
    noise=torch.Tensor(N_templates,N_nodes,N_nodes)
    torch.nn.init.normal_(Templates_features, mean=0.0012, std=2e-02)
    torch.nn.init.normal_(noise,mean=0,std=1e-2)
    return Templates+noise, Templates_features


class LTFGW(nn.Module):
    """ Layer for the local TFWG """
    def __init__(self, N_templates=10,N_templates_nodes=10,N_features=10,alpha0=None,train_node_weights=False):
        """
        N_features: number of node features
        N_templates: number of graph templates
        N_templates_nodes: number of nodes in each template
        alpha0: trade-off for the fused gromov-wasserstein distance. If None, alpha is optimised, else it is fixed at the given value.
        q0: weights on the nodes of the templates. If None, q0 is optimised, else it is fixed at the given value (must sum to 1 along the lines).
        """
        super().__init__()

        self.N_templates= N_templates
        self.N_templates_nodes=N_templates_nodes
        self.N_features=N_features
        self.Alphas=[]

        templates,templates_features=template_initialisation(self.N_templates_nodes,self.N_templates,self.N_features)
        self.templates=nn.Parameter(templates)
        self.templates_features = nn.Parameter(templates_features)

        if train_node_weights:
            q0=torch.Tensor(N_templates,N_templates_nodes)
            q0=torch.nn.init.uniform_(q0)
            self.q0=nn.Parameter(q0)
        else: 
            self.q0=torch.zeros(N_templates,N_templates_nodes)
            
        if alpha0 is None:
            alpha0=torch.Tensor([0])
            self.alpha0=nn.Parameter(alpha0)
        else:
            self.alpha0=torch.logit(alpha0)           

    def forward(self, x, edge_index):
        alpha=torch.sigmoid(self.alpha0)
        q=torch.softmax(self.q0,dim=1)
        x=distance_to_template(x,edge_index,self.templates_features,self.templates,alpha,q,2)
        return x



