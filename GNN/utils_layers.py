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
from ot.gromov import semirelaxed_fused_gromov_wasserstein2 
from ot.gromov import entropic_semirelaxed_fused_gromov_wasserstein
from torch_geometric.utils import subgraph as sub

from ot.backend import TorchBackend


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
    C = 0.5*(C + C.T)
    
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
        num_nodes=num_nodes,
        directed=False)
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
    #q = F.normalize(q, p=1, dim=1)

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
            M = ot.dist(x_sub, template_features)
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
                    M, C_sub, torch.Tensor(C_T[j]), p, qj, alpha=alpha, symmetric=True, max_iter=20)
            distances[i, j] = dist
    return distances


def semi_relaxed_marginals_to_template(
        x,
        edge_index,
        x_T,
        C_T,
        alpha,
        k,
        shortest_path,
        device,
        reg):
    """
    Computes the OT distance between each subgraphs of order k of G and the templates
    x : node features of the graph
    edge_index : edge indexes of the graph
    x_T : list of the node features of the templates
    C_T : list of the adjacency matrices of the templates
    alpha : trade-off parameter for fused gromov-wasserstein distance
    k : number of neighbours in the subgraphs
    shortest_path: bool
      If True, the shortest path matrix is used for the templates.
      Else, the adjacency matrix is used
    device: string
      Device, cuda or cpu
    reg: float
      If None, the semi-relaxed Qromov Wasserstein distance is used.
      Else, the entropic semi-relaxed Gromov Wasserstein is used, with the regularisation reg.
    """

    n = len(x)  # number of nodes in the graph
    n_T = len(x_T)  # number of templates
    n_feat = len(x[0])
    n_feat_T = len(x_T[0][0])
    n_templates_nodes=len(C_T[0][0])

    if not n_feat == n_feat_T:
        raise ValueError(
            'the templates and the graphs must have the same number of features')

    marginals = torch.zeros(n, n_T*n_templates_nodes)
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

        C_sub = graph_to_adjacency(
            n_sub,
            edges_sub,
            shortest_path,
            device=device)

        for j in range(n_T):

            template_features = x_T[j].reshape(
                len(x_T[j]), n_feat_T)  # reshape to use ot.dist
            M = ot.dist(x_sub, template_features)

            # cost matrix between the features of the subgraph and the template
            M = M.type(torch.float)

            # more normalization
            p = p / torch.sum(p)

            if reg==0:

                T = semirelaxed_fused_gromov_wasserstein2(M, C_sub, C_T[j], p, alpha=alpha,max_iter=20)

            else:

                T = entropic_semirelaxed_fused_gromov_wasserstein(M,C_sub, C_T[j], p, alpha=alpha,epsilon=reg, max_iter=20)

            q =torch.sum(T,0)
#            print(q)

            marginals[i, j*n_templates_nodes:(j+1)*n_templates_nodes] = q
    return marginals.to(device)



def FGW_one_node(M,C,p,alpha):
    nx = TorchBackend()
    T=torch.reshape(p,(len(p),1))
    fgw_dist=(1-alpha)*torch.matmul(p,M)+alpha*torch.sum(C**2)
    C2=torch.tensor([[0.]])
    fgw_dist = nx.set_gradients(fgw_dist, (alpha,M,C,p),
                                        (torch.sum(C**2)-torch.matmul(p,M),(1-alpha)*T,2 * C * nx.outer(p, p) - 2 * nx.dot(T, nx.dot(C2, T.T)),(1-alpha)*M))
    return fgw_dist


def distance_to_template_one_node(
        x,
        edge_index,
        x_T,
        C_T,
        alpha,
        q,
        k,
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

            dist = FGW_one_node(M,C_sub,p,alpha)
            distances[i, j] = dist
    return distances

def distance_to_templates3(G_edges, tplt_adjacencies, G_features, tplt_features, tplt_weights, alpha, multi_alpha,batch=None):
    """
    Computes the FGW distances between a graph and graph templates.

    Parameters
    ----------
    G_edges : torch tensor, shape(n_edges, 2)
        Edge indexes of the graph in the Pytorch Geometric format.
    tplt_adjacencies : list of torch tensors, shape (n_templates, n_template_nodes, n_templates_nodes)
        List of the adjacency matrices of the templates.
    G_features : torch tensor, shape (n_nodes, n_features)
        Node features of the graph.
    tplt_features : list of torch tensors, shape (n_templates, n_template_nodes, n_features)
        List of the node features of the templates.
    weights : torch tensor, shape (n_templates, n_template_nodes)
        Weights on the nodes of the templates.
    alpha : float
        Trade-off parameter (0 < alpha < 1).
        Weights features (alpha=0) and structure (alpha=1).
    multi_alpha: bool, optional
        If True, the alpha parameter is a vector of size n_templates.
    batch: torch tensor
        Node level batch vector.

    Returns
    -------
    distances : torch tensor, shape (n_templates)
        Vector of fused Gromov-Wasserstein distances between the graph and the templates.
    """
    
    if not batch==None:
      n_T, _, n_feat_T = tplt_features.shape

      num_graphs=torch.max(batch)+1
      distances=torch.zeros(num_graphs,n_T)
      
      #iterate over the graphs in the batch
      for i in range(num_graphs):
        
        nodes=torch.where(batch==i)[0]

        G_edges_i,_=sub(nodes,edge_index=G_edges,relabel_nodes=True)
        G_features_i=G_features[nodes]

        n, n_feat = G_features_i.shape

        weights_G = torch.ones(n) / n

        C = torch.sparse_coo_tensor(G_edges_i, torch.ones(len(G_edges_i[0])), size=(n, n)).type(torch.float)
        C = C.to_dense()

        if not n_feat == n_feat_T:
            raise ValueError('The templates and the graphs must have the same feature dimension.')

        for j in range(n_T):

            template_features = tplt_features[j].reshape(len(tplt_features[j]), n_feat_T)
            M = ot.dist(G_features_i, template_features).type(torch.float)

            p_nump = weights_G.numpy()
            p_nump = np.asarray(p_nump, dtype=np.float64)
            sum_p = p_nump.sum(0)
            q_nump = tplt_weights[j].detach().numpy()
            q_nump = np.asarray(q_nump, dtype=np.float64)
            sum_q = q_nump.sum(0)
            if not abs(sum_q - sum_p) < np.float64(1.5 * 10**(-7)):
                if sum_q > sum_p:
                    weights_G[0] += abs(sum_q - sum_p)
                else:
                    weights_G[0] -= abs(sum_q - sum_p)     


            if multi_alpha:
                embedding = ot.fused_gromov_wasserstein2(M, C, tplt_adjacencies[j], weights_G, tplt_weights[j], alpha=alpha[j], symmetric=True, max_iter=50)
            else:
                embedding = ot.fused_gromov_wasserstein2(M, C, tplt_adjacencies[j], weights_G, tplt_weights[j], alpha=alpha, symmetric=True, max_iter=50)

            distances[i,j] = embedding

    else:
         
      n, n_feat = G_features.shape
      n_T, _, n_feat_T = tplt_features.shape

      weights_G = torch.ones(n) / n

      C = torch.sparse_coo_tensor(G_edges, torch.ones(len(G_edges[0])), size=(n, n)).type(torch.float)
      C = C.to_dense()

      if not n_feat == n_feat_T:
          raise ValueError('The templates and the graphs must have the same feature dimension.')

      distances = torch.zeros(n_T)

      for j in range(n_T):

          template_features = tplt_features[j].reshape(len(tplt_features[j]), n_feat_T)
          M = ot.dist(G_features, template_features).type(torch.float)

          if multi_alpha:
              embedding = ot.fused_gromov_wasserstein2(M, C, tplt_adjacencies[j], weights_G, tplt_weights[j], alpha=alpha[j], symmetric=True, max_iter=100)
          else:
              embedding = ot.fused_gromov_wasserstein2(M, C, tplt_adjacencies[j], weights_G, tplt_weights[j], alpha=alpha, symmetric=True, max_iter=100)

          distances[j] = embedding

    return distances