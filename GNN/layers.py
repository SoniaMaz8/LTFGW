import torch
import torch.nn as nn
from GNN.utils_layers import *
import torch.nn.functional as F


def template_initialisation(n_nodes, n_templates, n_features, mean, std, template_sizes):
    """"
    Function that initialises templates for the LTFGW layer
    Input:
      n_templates: number of templates
      n_nodes: number of nodes per template
      n_features: number of features for the nodes
    Output:
      Templates: list of the adjancecy matrices of the templates
      Templates_features: list of the features of the nodes of each template
    """
    if template_sizes==None:
        Templates = torch.rand((n_templates, n_nodes, n_nodes))
        Templates_features = torch.Tensor(n_templates, n_nodes, n_features)
        torch.nn.init.normal_(Templates_features, mean=mean, std=std)
        q0 = torch.zeros(n_templates, n_nodes)
        return 0.5 * (Templates + torch.transpose(Templates, 1, 2)
                    ), Templates_features, q0
    
    else:
        Templates=[]
        Templates_features=[]
        q0=[]
        for size in template_sizes:
            template=torch.rand((size, size))
            template_feature = torch.Tensor(size, n_features)
            torch.nn.init.normal_(template_feature, mean=mean, std=std)
            q=torch.zeros(size)
            q0.append(q)
            Templates.append(0.5*(template+torch.transpose(template,0,1)))
            Templates_features.append(template_feature)
             
        return Templates,Templates_features, q0
    




class LTFGW_log(nn.Module):
    """ Layer for the local TFWG """

    def __init__(
            self,
            n_nodes,
            n_templates=10,
            n_templates_nodes=10,
            n_features=10,
            k=1,
            alpha0=None,
            mean_init=0,
            std_init=0.001,
            train_node_weights=True,
            shortest_path=False,
            template_sizes=None):
        """
        n_features: number of node features
        n_templates: number of graph templates
        n_templates_nodes: number of nodes in each template
        alpha0: trade-off for the fused gromov-wasserstein distance. If None, alpha is optimised, else it is fixed at the given value.
        q0: weights on the nodes of the templates. If None, q0 is optimised, else it is fixed at the given value (must sum to 1 along the lines).
        local alpha: wether to learn one tradeoff parameter for the FGW for each node or for the whole graph
        """
        super().__init__()

        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.n_features = n_features
        self.k = k
        self.shortest_path = shortest_path
        self.template_sizes=template_sizes

        templates, templates_features = template_initialisation(
            self.n_templates_nodes, self.n_templates, self.n_features, mean_init, std_init,template_sizes)
        
        if self.template_sizes==None:
            self.templates = nn.Parameter(templates)
            self.templates_features = nn.Parameter(templates_features)
        
        else:
            self.templates = nn.ParameterList(templates)
            self.templates_features = nn.ParameterList(templates_features)
            
        self.softmax = nn.Softmax(dim=1)
                
        if train_node_weights:
            q0 = torch.zeros(n_templates, n_templates_nodes)
            self.q0 = nn.Parameter(q0)
        else:
            self.q0 = torch.zeros(n_templates, n_templates_nodes)

        # initialize the tradeoff parameter alpha
        if alpha0 is None:
                alpha0 = torch.Tensor([0])
                self.alpha0 = nn.Parameter(alpha0)
        else:
                alpha0 = torch.zeros([alpha0])
                self.alpha0 = torch.logit(alpha0)

    def forward(self, x, edge_index):
        alpha = torch.sigmoid(self.alpha0)
        q = self.softmax(self.q0)
        x = torch.log(
            distance_to_template(
                x,
                edge_index,
                self.templates_features,
                self.templates,
                alpha,
                q,
                self.k,
                self.local_alpha,
                self.shortest_path))
        return x

class LTFGW(nn.Module):
    """ Layer for the local TFWG """

    def __init__(
            self,
            n_nodes,
            n_templates=10,
            n_templates_nodes=10,
            n_features=10,
            k=1,
            alpha0=None,
            mean_init=0,
            std_init=0.001,
            train_node_weights=True,
            shortest_path=False,
            template_sizes=None):
        """
        n_features: number of node features
        n_templates: number of graph templates
        n_templates_nodes: number of nodes in each template
        alpha0: trade-off for the fused gromov-wasserstein distance. If None, alpha is optimised, else it is fixed at the given value.
        q0: weights on the nodes of the templates. If None, q0 is optimised, else it is fixed at the given value (must sum to 1 along the lines).
        local alpha: wether to learn one tradeoff parameter for the FGW for each node or for the whole graph
        """
        super().__init__()

        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.n_features = n_features
        self.k = k
        self.shortest_path = shortest_path
        self.template_sizes=template_sizes


        templates, templates_features, q0 = template_initialisation(
        self.n_templates_nodes, self.n_templates, self.n_features, mean_init, std_init, template_sizes)

        if self.template_sizes==None:
            self.templates = nn.Parameter(templates)
            self.templates_features = nn.Parameter(templates_features)
            if train_node_weights:
                 self.q0=torch.nn.Parameter(q0)
        
        else:
            self.templates = nn.ParameterList(templates)
            self.templates_features = nn.ParameterList(templates_features)
            if  train_node_weights :
                  self.q0=torch.nn.ParameterList(q0)
        
        
        self.softmax1 = nn.Softmax(dim=1)  
        self.softmax2 = nn.Softmax(dim=0) 

        # initialize the tradeoff parameter alpha
        if alpha0 is None:
                alpha0 = torch.Tensor([0])
                self.alpha0 = nn.Parameter(alpha0)
        else:
                alpha0 = torch.tensor([alpha0])
                self.alpha0 = torch.logit(alpha0)

    def forward(self, x, edge_index):
        alpha = torch.sigmoid(self.alpha0)

        if self.template_sizes==None:
          q = self.softmax1(self.q0)
        else: 
          q=[]
          for i in range(self.n_templates):
            q.append(self.softmax2(self.q0[i]))
               
        x = distance_to_template(
            x,
            edge_index,
            self.templates_features,
            self.templates,
            alpha,
            q,
            self.k,
            self.shortest_path)
        return x

        
class LTFGW_semirelaxed(nn.Module):
    """ Layer for the local TFWG """

    def __init__(
            self,
            n_nodes,
            n_templates=10,
            n_templates_nodes=10,
            n_features=10,
            k=1,
            mean_init=0,
            std_init=0.001,
            alpha0=None,
            shortest_path=False,
            device='cpu',
            template_sizes=None):
        """
        n_features: number of node features
        n_templates: number of graph templates
        n_templates_nodes: number of nodes in each template
        alpha0: trade-off for the fused gromov-wasserstein distance. If None, alpha is optimised, else it is fixed at the given value.
        q0: weights on the nodes of the templates. If None, q0 is optimised, else it is fixed at the given value (must sum to 1 along the lines).
        local alpha: wether to learn one tradeoff parameter for the FGW for each node or for the whole graph
        """
        super().__init__()

        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.n_features = n_features
        self.k = k
        self.shortest_path = shortest_path
        self.device = device

        templates, templates_features = template_initialisation(
            self.n_templates_nodes, self.n_templates, self.n_features, mean_init, std_init, template_sizes)
        self.templates = nn.Parameter(templates)
        self.templates_features = nn.Parameter(templates_features)
        self.softmax = nn.Softmax(dim=1)

        # initialize the tradeoff parameter alpha
        if alpha0 is None:
                alpha0 = torch.Tensor([0])
                self.alpha0 = nn.Parameter(alpha0)
        else:
                alpha0 = torch.Tensor([alpha0])
                self.alpha0 = torch.logit(alpha0)

    def forward(self, x, edge_index):
        alpha = torch.sigmoid(self.alpha0)
        x = distance_to_template_semirelaxed(
            x,
            edge_index,
            self.templates_features,
            self.templates,
            alpha,
            self.k,
            self.shortest_path,
            self.device)
        return x



class LTFGW_one_node(nn.Module):
    """ Layer for the local TFWG """

    def __init__(
            self,
            n_nodes,
            n_templates=10,
            n_templates_nodes=10,
            n_features=10,
            k=1,
            alpha0=None,
            mean_init=0,
            std_init=0.001,
            train_node_weights=True,
            shortest_path=False):
        """
        n_features: number of node features
        n_templates: number of graph templates
        n_templates_nodes: number of nodes in each template
        alpha0: trade-off for the fused gromov-wasserstein distance. If None, alpha is optimised, else it is fixed at the given value.
        q0: weights on the nodes of the templates. If None, q0 is optimised, else it is fixed at the given value (must sum to 1 along the lines).
        local alpha: wether to learn one tradeoff parameter for the FGW for each node or for the whole graph
        """
        super().__init__()

        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.n_features = n_features
        self.k = k
        self.shortest_path = shortest_path

        templates, templates_features = template_initialisation(
            self.n_templates_nodes, self.n_templates, self.n_features, mean_init, std_init)
        self.templates = nn.Parameter(templates)
        self.templates_features = nn.Parameter(templates_features)
        self.softmax = nn.Softmax(dim=1)

        if train_node_weights:
            q0 = torch.zeros(n_templates, n_templates_nodes)
            self.q0 = nn.Parameter(q0)
        else:
            self.q0 = torch.zeros(n_templates, n_templates_nodes)

        # initialize the tradeoff parameter alpha
        if alpha0 is None:
                alpha0 = torch.Tensor([0])
                self.alpha0 = nn.Parameter(alpha0)
        else:
                alpha0 = torch.tensor([alpha0])
                self.alpha0 = torch.logit(alpha0)

    def forward(self, x, edge_index):
        alpha = torch.sigmoid(self.alpha0)
        q = self.softmax(self.q0)
        x = distance_to_template_one_node(
            x,
            edge_index,
            self.templates_features,
            self.templates,
            alpha,
            q,
            self.k,
            self.local_alpha,
            self.shortest_path)
        return x
    
