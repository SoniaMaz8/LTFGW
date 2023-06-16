import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
import ot
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch_geometric.utils import k_hop_subgraph
from scipy.sparse.csgraph import shortest_path as function_shortest_path
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data as GraphData
from ot.gromov import semirelaxed_fused_gromov_wasserstein2 as semirelaxed_fgw
import time


def adjacency_to_graph(C, F):
    """
    Returns a torch_geometric graph given binary adjacency matrix and features
    C : adjacency matrix of the graph
    F : features of the nodes of the graph
    """
    edges = torch.where(C == 1)
    edge_index = torch.stack(edges)
    return GraphData(x=F, edge_index=edge_index)


def graph_to_adjacency(n, edges, shortest_path, device='cpu'):
    """"
    adjacency matrix of a graph given its nodes and edges in a torch.geometric format
    n : number of nodes
    edges : edges in the format [[senders],[receivers]]

    Returns: sparse adjacency matrix C
    """
    ones = torch.ones(len(edges[0])).to(device)
    C = torch.sparse_coo_tensor(edges, ones, size=(n, n))
    C = C.to_dense()
    C = C + C.T
    if not shortest_path:
        return C
    else:
        graph = csr_matrix(C)
        dist_matrix = function_shortest_path(graph)
        return torch.Tensor(dist_matrix).to(device)


def subgraph(x, edge_index, node_idx, order, num_nodes):
    """
    Computes the edges and nodes of a subgraph center at node_idx of order k
    C : adjacency matrix of the graph
    x : features of the nodes of the graph
    node_idx : index of the node to center the subgraph
    order : order of te subgraph (number of neigbours)
    """

    sub_G = k_hop_subgraph(
        node_idx,
        order,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes)
    x_sub = x[sub_G[0]]
    edges_sub = sub_G[1]
    central_node_index = sub_G[2]
    return x_sub, edges_sub, central_node_index


def distance_to_template(
        x,
        edge_index,
        x_T,
        C_T,
        alpha,
        q,
        k,
        local_alpha,
        shortest_path):
    """
    Computes the OT distance between each subgraphs of order k of G and the templates
    x : node features of the graph
    edge_index : edge indexes of the graph
    x_T : list of the node features of the templates
    C_T : list of the adjacency matrices of the templates
    alpha : trade-off parameter for fused gromov-wasserstein distance
    k : number of neighbours in the subgraphs
    """

    n = len(x)  # number of nodes in the graph
    n_T = len(x_T)  # number of templates
    n_feat = len(x[0])
    n_feat_T = len(x_T[0][0])

    # normalize q for gromov-wasserstein
    q = F.normalize(q, p=1, dim=1)

    if not n_feat == n_feat_T:
        raise ValueError(
            'the templates and the graphs must have the same number of features')

    distances = torch.zeros(n, n_T)
    for i in range(n):
        x_sub, edges_sub, central_node_index = subgraph(x, edge_index, i, k, n)
        # reshape pour utiliser ot.dist
        x_sub = x_sub.reshape(len(x_sub), n_feat)
        n_sub = len(x_sub)

        if n_sub > 1:  # more weight on central node
            val = (1 - (k + 1) / (k + 2)) / (n_sub - 1)
            p = torch.ones(n_sub) * val
            p[central_node_index] = (k + 1) / (k + 2)
            p = F.normalize(p, p=1, dim=0)  # normalize for gromov-wasserstein

        else:  # if the node is isolated
            p = torch.ones(1)
            # normalize p for gromov-wasserstein
            p = F.normalize(p, p=1, dim=0)

        C_sub = graph_to_adjacency(
            n_sub, edges_sub, shortest_path).type(
            torch.float)

        for j in range(n_T):

            template_features = x_T[j].reshape(
                len(x_T[j]), n_feat_T)  # reshape pour utiliser ot.dist
            M = ot.dist(x_sub, template_features).clone(
            ).detach().requires_grad_(True)
            # cost matrix between the features of the subgraph and the template
            M = M.type(torch.float)

            # more normalization
            qj = q[j] / torch.sum(q[j])
            p = p / torch.sum(p)

            # ensure that p and q have the same sum
            p_nump = p.numpy()
            p_nump = np.asarray(p_nump, dtype=np.float64)
            sum_p = p_nump.sum(0)
            q_nump = qj.detach().numpy()
            q_nump = np.asarray(q_nump, dtype=np.float64)
            sum_q = q_nump.sum(0)
            if not abs(sum_q - sum_p) < np.float64(1.5 * 10**(-7)):
                if sum_q > sum_p:
                    p[0] += abs(sum_q - sum_p)
                else:
                    qj[0] += abs(sum_q - sum_p)

            dist = ot.gromov.fused_gromov_wasserstein2(
                    M, C_sub, C_T[j], p, qj, alpha=alpha, symmetric=True, max_iter=20)
            distances[i, j] = dist
    return distances


def distance_to_template_semirelaxed(
        x,
        edge_index,
        x_T,
        C_T,
        alpha,
        k,
        shortest_path,
        device):
    """
    Computes the OT distance between each subgraphs of order k of G and the templates
    x : node features of the graph
    edge_index : edge indexes of the graph
    x_T : list of the node features of the templates
    C_T : list of the adjacency matrices of the templates
    alpha : trade-off parameter for fused gromov-wasserstein distance
    k : number of neighbours in the subgraphs
    """

    n = len(x)  # number of nodes in the graph
    n_T = len(x_T)  # number of templates
    n_feat = len(x[0])
    n_feat_T = len(x_T[0][0])

    if not n_feat == n_feat_T:
        raise ValueError(
            'the templates and the graphs must have the same number of features')

    distances = torch.zeros(n, n_T)
    for i in range(n):
        x_sub, edges_sub, central_node_index = subgraph(x, edge_index, i, k, n)
        # reshape pour utiliser ot.dist
        x_sub = x_sub.reshape(len(x_sub), n_feat)
        n_sub = len(x_sub)

        if n_sub > 1:  # more weight on central node
            val = (1 - (k + 1) / (k + 2)) / (n_sub - 1)
            p = torch.ones(n_sub) * val
            p = p.to(device)
            p[central_node_index] = (k + 1) / (k + 2)
            p = F.normalize(p, p=1, dim=0)  # normalize for gromov-wasserstein

        else:  # if the node is isolated
            p = torch.ones(1)
            # normalize p for gromov-wasserstein
            p = F.normalize(p, p=1, dim=0)
            p = p.to(device)

        p = p.type(torch.double)
        C_sub = graph_to_adjacency(
            n_sub,
            edges_sub,
            shortest_path,
            device=device).type(
            torch.double)

        for j in range(n_T):

            template_features = x_T[j].reshape(
                len(x_T[j]), n_feat_T)  # reshape pour utiliser ot.dist
            M = ot.dist(x_sub, template_features).clone(
            ).detach().requires_grad_(True)
            # cost matrix between the features of the subgraph and the template
            M = M.type(torch.float)

            # more normalization
            p = p / torch.sum(p)

            dist = semirelaxed_fgw(
                    M,
                    C_sub,
                    C_T[j].type(
                        torch.double),
                    p,
                    alpha=alpha,
                    symmetric=True,
                    max_iter=20)
            distances[i, j] = dist
    return distances.to(device)
