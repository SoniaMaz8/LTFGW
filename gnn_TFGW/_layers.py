"""
Template Fused Gromov Wasserstein
"""

import torch
import torch.nn as nn
from ._utils import TFGW_template_initialisation, FGW_pooling


class TFGWPooling(nn.Module):
    """
    Template Fused Gromov-Wasserstein (TFGW) layer. This layer is a pooling layer for graph neural networks.
        It computes the fused Gromov-Wasserstein distances between the graph and a set of templates.

    Parameters
    ----------
    n_features : int
        Feature dimension of the nodes.
    n_templates : int
         Number of graph templates.
    n_templates_nodes : int
        Number of nodes in each template.
    alpha0 : float, optional
        FGW trade-off parameter (0 < alpha < 1). If None alpha is trained, else it is fixed at the given value.
        Weights features (alpha=0) and structure (alpha=1).
    train_node_weights : bool, optional
        If True, the templates node weights are learned.
        Else, they are uniform.
    multi_alpha: bool, optional
        If True, the alpha parameter is a vector of size n_templates.


    References
    ----------
    .. [52]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Template based graph neural network with optimal transport distances"
    """

    def __init__(self, n_features, n_templates=2, n_template_nodes=2, alpha0=None, train_node_weights=True, multi_alpha=False,mean=0,std=1):
        """
        Template Fused Gromov-Wasserstein (TFGW) layer. This layer is a pooling layer for graph neural networks.
            It computes the fused Gromov-Wasserstein distances between the graph and a set of templates.

        Parameters
        ----------
        n_features : int
                Feature dimension of the nodes.
        n_templates : int
                Number of graph templates.
        n_templates_nodes : int
                Number of nodes in each template.
        alpha : float, optional
                Trade-off parameter (0 < alpha < 1). If None alpha is trained, else it is fixed at the given value.
                Weights features (alpha=0) and structure (alpha=1).
        train_node_weights : bool, optional
                If True, the templates node weights are learned.
                Else, they are uniform.
        multi_alpha: bool, optional
                If True, the alpha parameter is a vector of size n_templates.

        References
        ----------
        .. [52]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
              "Template based graph neural network with optimal transport distances"

        """
        super().__init__()

        self.n_templates = n_templates
        self.n_templates_nodes = n_template_nodes
        self.n_features = n_features
        self.multi_alpha = multi_alpha
        
        templates, templates_features, self.q0 = TFGW_template_initialisation( self.n_templates, self.n_templates_nodes, self.n_features, mean,std)
        self.templates = nn.Parameter(templates)
        self.template_features = nn.Parameter(templates_features)

        self.softmax = nn.Softmax(dim=1)

        if train_node_weights:
            self.q0 = nn.Parameter(self.q0)

        if alpha0 is None:
            if multi_alpha:
                self.alpha0 =torch.Tensor([0] * self.n_templates)
            else:
                alpha0 = torch.Tensor([0])
            self.alpha0 = nn.Parameter(alpha0)
        else:
            if multi_alpha:
                self.alpha0 = torch.Tensor([alpha0] * self.n_templates)
                self.alpha0 = nn.Parameter(self.alpha0) 
            else:
                self.alpha0 = torch.Tensor([alpha0])
            self.alpha0 = torch.logit(alpha0)

    def forward(self, x, edge_index):
        alpha = torch.sigmoid(self.alpha0)
        q = self.softmax(self.q0)
        x = FGW_pooling(edge_index, self.templates, x, self.template_features, q, alpha, self.multi_alpha)
        return x