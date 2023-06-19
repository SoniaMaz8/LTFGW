import torch
import torch.nn as nn
from GNN.utils_layers import distance_to_template_semirelaxed, distance_to_template
import torch.nn.functional as F


def template_initialisation(n_nodes, n_templates, n_features, mean, std):
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
    Templates = torch.rand((n_templates, n_nodes, n_nodes))
    Templates_features = torch.Tensor(n_templates, n_nodes, n_features)
    torch.nn.init.normal_(Templates_features, mean=mean, std=std)
    return 0.5 * (Templates + torch.transpose(Templates, 1, 2)
                  ), Templates_features


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
            local_alpha=True,
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

        self.local_alpha = local_alpha

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
            local_alpha=True,
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

        self.local_alpha = local_alpha

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
        x = distance_to_template(
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
            local_alpha=False,
            shortest_path=False,
            device='cpu'):
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

        self.local_alpha = local_alpha

        templates, templates_features = template_initialisation(
            self.n_templates_nodes, self.n_templates, self.n_features, mean_init, std_init)
        self.templates = nn.Parameter(templates)
        self.templates_features = nn.Parameter(templates_features)
        self.softmax = nn.Softmax(dim=1)

        # initialize the tradeoff parameter alpha
        if alpha0 is None:
            if self.local_alpha:
                alpha0 = torch.zeros(n_nodes)
                self.alpha0 = nn.Parameter(alpha0)
            else:
                alpha0 = torch.Tensor([0])
                self.alpha0 = nn.Parameter(alpha0)
        else:
            if self.local_alpha:
                alpha0 = torch.ones(n_nodes) * alpha0
                self.alpha0 = torch.logit(alpha0)
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
            self.local_alpha,
            self.shortest_path,
            self.device)
        return x
