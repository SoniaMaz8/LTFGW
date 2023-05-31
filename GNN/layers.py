import torch
import torch.nn as nn
from GNN.utils import distance_to_template
from torch_geometric.data import Data as GraphData
import torch.nn.functional as F
import pandas as pd
from torch_geometric.utils import subgraph


def template_initialisation(N_nodes,N_templates,N_features):
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
    torch.nn.init.normal_(Templates_features, mean=0.005, std=0.0067)
    torch.nn.init.normal_(noise,mean=0,std=1e-2)
    Templates=Templates+noise
    return 0.5*(Templates+torch.transpose(Templates,1,2)), Templates_features


class LTFGW(nn.Module):
    """ Layer for the local TFWG """
    def __init__(self,n_nodes, n_templates=10,n_templates_nodes=10,n_features=10,k=1,alpha0=None,train_node_weights=True,local_alpha=True,shortest_path=False):
        """
        n_features: number of node features
        n_templates: number of graph templates
        n_templates_nodes: number of nodes in each template
        alpha0: trade-off for the fused gromov-wasserstein distance. If None, alpha is optimised, else it is fixed at the given value.
        q0: weights on the nodes of the templates. If None, q0 is optimised, else it is fixed at the given value (must sum to 1 along the lines).
        local alpha: wether to learn one tradeoff parameter for the FGW for each node or for the whole graph 
        """
        super().__init__()

        self.n_templates= n_templates
        self.n_templates_nodes=n_templates_nodes
        self.n_features=n_features
        self.k=k
        self.shortest_path=shortest_path

        self.local_alpha=local_alpha

        templates,templates_features=template_initialisation(self.n_templates_nodes,self.n_templates,self.n_features)
        self.templates=nn.Parameter(templates)
        self.templates_features = nn.Parameter(templates_features)
        self.softmax=nn.Softmax(dim=1)

        if train_node_weights:
            q0=torch.zeros(n_templates,n_templates_nodes)
            self.q0=nn.Parameter(q0)
        else: 
            self.q0=torch.zeros(n_templates,n_templates_nodes)
            
        #initialize the tradeoff parameter alpha
        if alpha0 is None:
            if self.local_alpha:
                alpha0=torch.zeros(n_nodes)
                self.alpha0=nn.Parameter(alpha0)
            else:
                alpha0=torch.Tensor([0])
                self.alpha0=nn.Parameter(alpha0)
        else:
            if self.local_alpha:
                alpha0=torch.ones(n_nodes)*alpha0
                self.alpha0=torch.logit(alpha0) 
            else:
                alpha0=torch.zeros(n_nodes)
                self.alpha0=torch.logit(alpha0) 

    def forward(self, x, edge_index):
        alpha=torch.sigmoid(self.alpha0)
        q=self.softmax(self.q0)
        x=distance_to_template(x,edge_index,self.templates_features,self.templates,alpha,q,self.k,self.local_alpha,self.shortest_path)
        return x



