import torch
import torch.nn as nn
from utils import distance_to_template
from torch_geometric.data import Data as GraphData


dataset=torch.load('Citeseer_data/graph_Citeseer.pt')

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
    def __init__(self, N_templates=15,N_templates_nodes=10,N_features=10, alpha_is_param=False):
        """
        N_features: number of node features
        N_templates: number of graph templates
        N_templates_nodes: number of nodes in each template
        """
        super().__init__()

        self.N_templates= N_templates
        self.N_templates_nodes=N_templates_nodes
        self.N_features=N_features
        self.alpha_is_param=alpha_is_param
        
        #templates initilisation as subgraphs of the dataset after one GCN layer

        templates,templates_features=template_initialisation(self.N_templates_nodes,self.N_templates,self.N_features)
        self.templates=nn.Parameter(templates)
        self.templates_features = nn.Parameter(templates_features)
        
        if alpha_is_param:
          alpha=torch.Tensor(0)
          torch.nn.init.unif(alpha)
          self.alpha=nn.Parameter(alpha)
        else:
            self.alpha=0.5*torch.ones(1)

    def forward(self, x, edge_index):
        x=distance_to_template(x,edge_index,self.templates_features,self.templates,self.alpha)
        return x




