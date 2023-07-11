# -*- coding: utf-8 -*-
"""
GNN layers utils
"""

import torch
import ot 
from ot import dist
from ot.gromov import fused_gromov_wasserstein2
import numpy as np


def TFGW_template_initialisation(n_templates, n_template_nodes, n_features, feature_init_mean=0., feature_init_std=1.):
    """"
    Initialises templates for the Template Fused Gromov Wasserstein layer.
    Returns the adjacency matrices and the features of the nodes of the templates.
    Adjacency matrics are intialised uniformly with values in :math:[0,1]
    Features of the nodes are intialised following a normal distribution.

    Parameters
    ----------

      n_templates: int
        Number of templates.
      n_template_nodes: int
        Number of nodes per template.
      n_features: int
        Number of features for the nodes.
      feature_init_mean: float, optional
        Mean of the random normal law to initialize the template features.
      feature_init_std: float, optional
        Standard deviation of the random normal law to initialize the template features.

    Returns
    ----------
      Templates: torch tensor, shape (n_templates, n_template_nodes, n_template_nodes)
           Adjancency matrices for the templates.
      Templates_features: torch tensor, shape (n_templates, n_template_nodes, n_features)
           Node features for each template.
      q0: weight on the template nodes.
    """

    templates_adjacency = torch.rand((n_templates, n_template_nodes, n_template_nodes))
    templates_features = torch.Tensor(n_templates, n_template_nodes, n_features)

    torch.nn.init.normal_(templates_features, mean=feature_init_mean, std=feature_init_std)

    templates_adjacency = templates_adjacency

    q0 = torch.zeros(n_templates, n_template_nodes)

    return 0.5 * (templates_adjacency + torch.transpose(templates_adjacency, 1, 2)), templates_features, q0


def FGW_pooling(edges_G, C_T, x_G, x_T, weights, alpha, multi_alpha):
    """
    Computes the FGW distances between a graph and graph templates.

    Parameters
    ----------
    edges_G : torch tensor, shape(n_edges, 2)
        Edge indexes of the graph in the Pytorch Geometric format.
    C_T : list of torch tensors, shape (n_templates, n_template_nodes, n_templates_nodes)
        List of the adjacency matrices of the templates.
    x_G : torch tensor, shape (n_nodes, n_features)
        Node features of the graph.
    x_T : list of torch tensors, shape (n_templates, n_template_nodes, n_features)
        List of the node features of the templates.
    weights : torch tensor, shape (n_templates, n_template_nodes)
        Weights on the nodes of the templates.
    alpha : float
        Trade-off parameter (0 < alpha < 1).
        Weights features (alpha=0) and structure (alpha=1).
    multi_alpha: bool, optional
        If True, the alpha parameter is a vector of size n_templates.

    Returns
    -------
    distances : torch tensor, shape (n_templates)
        Vector of fused Gromov-Wasserstein distances between the graph and the templates.
    """

    n, n_feat = x_G.shape
    n_T, _, n_feat_T = x_T.shape

    weights_p = torch.ones(n) / n

    C = torch.sparse_coo_tensor(edges_G, torch.ones(len(edges_G[0])), size=(n, n)).type(torch.float)
    C = C.to_dense()

    if not n_feat == n_feat_T:
        raise ValueError('The templates and the graphs must have the same feature dimension.')

    distances = torch.zeros(n_T)

    for j in range(n_T):

        qj=weights[j]
        # ensure that p and q have the same sum
        p_nump = weights_p.numpy()
        p_nump = np.asarray(p_nump, dtype=np.float64)
        sum_p = p_nump.sum(0)
        q_nump = qj.detach().numpy()
        q_nump = np.asarray(q_nump, dtype=np.float64)
        sum_q = q_nump.sum(0)
        if not abs(sum_q - sum_p) < np.float64(1.5 * 10**(-7)):
            if sum_q > sum_p:
                 weights_p[0] += abs(sum_q - sum_p)
            else:
                weights_p[0] -= abs(sum_q - sum_p)  

        template_features = x_T[j].reshape(len(x_T[j]), n_feat_T)
        M = dist(x_G, template_features).type(torch.float)

        if multi_alpha:
            embedding = fused_gromov_wasserstein2(M, C, C_T[j], weights_p, qj, alpha=alpha[j], symmetric=True, max_iter=100)
        else:
            embedding = fused_gromov_wasserstein2(M, C, C_T[j], weights_p, qj, alpha=alpha, symmetric=True, max_iter=100)
        distances[j] = embedding

    return distances
