import torch
import torch.nn as nn
from GNN.utils_layers import *
import torch.nn.functional as F


def template_initialisation(n_nodes, n_templates, n_features, mean, std, template_sizes=None):
    """"
    Function that initialises templates for the LTFGW layer
    Input:
      n_templates: int
        Number of templates.
      n_nodes: int
        Number of nodes per template.
      n_features: int
        Number of features for the nodes.
      mean: float
        Mean of the random normal law to initialize the template features.
      std: float
        Std of the random normal law to initialize the template features.
      template_sizes: bool
         If None, all template have the same number of nodes. 
         Else, list of the number of nodes of the templates. 
    Output:
      Templates: list of the adjancecy matrices of the templates
      Templates_features: list of the features of the nodes of each template
      q0: list of weights of the templates
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
    


class LTFGW(nn.Module):
    """ Layer for the local TFWG 

    Computes an embedding for each node by computing a 1 dimensional array of distances between learned templates
      and the node's neighbourhood.

    The distance used is the fused Gromov-Wasserstein distance.

    n_templates: int, optional
        Number of graph templates.
    n_templates_nodes: int, optional
        Number of nodes in each template.
    n_features: int, optional
        Number of node features.
    k: int, optional
        Number of hops fot he nodes' neighbourhood.
    mean_init: float, optional
        Mean of the random normal law to initialize the template features.
    std_init:  float, optional
        Std of the random normal law to initialize the template features.
    train_node_weights: bool, optional
        If True, the node weights are trained.
        Else they are uniform.
    shortest_path: bool, optional
        If True, the templates are characterized by their shortest path matrix.
        Else, the adjacency matrix is used.
    template_sizes: if None, all template have the same number of nodes. 
        Else, list of the number of nodes of the templates. 
    log: bool
       If True the log of the output of the layer is used.
    
    """
    def __init__(
            self,
            n_templates=10,
            n_templates_nodes=10,
            n_features=10,
            k=1,
            alpha0=None,
            mean_init=0,
            std_init=0.001,
            train_node_weights=True,
            shortest_path=False,
            template_sizes=None,
            log=False):
        """ Layer for the local TFWG 

        Computes an embedding for each node by computing a 1 dimensional array of distances between learned templates
        and the node's neighbourhood.

        The distance used is the fused Gromov-Wasserstein distance.

        Parameters
        ----------

        n_templates: int, optional
          Number of graph templates.
        n_templates_nodes: int, optional
          Number of nodes in each template.
        n_features: int, optional
          Number of node features.
        k: int, optional
          Number of hops fot he nodes' neighbourhood.
        mean_init: float, optional
          Mean of the random normal law to initialize the template features.
        std_init:  float, optional
          Std of the random normal law to initialize the template features.
        train_node_weights: bool, optional
          If True, the node weights are trained.
          Else they are uniform.
        shortest_path: bool, optional
          If True, the templates are characterized by their shortest path matrix.
          Else, the adjacency matrix is used.
        template_sizes: if None, all template have the same number of nodes. 
          Else, list of the number of nodes of the templates. 
        log: bool,optional
          If True, the output is the log of the LTFGW layer.
          Else, it is the LTFGW layer.
        """
        super().__init__()

        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.n_features = n_features
        self.k = k
        self.shortest_path = shortest_path
        self.template_sizes=template_sizes
        self.log=log

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
        
        

        self.softmax_weights = nn.Softmax(dim=1)  
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
        templates_softmax=torch.softmax(self.templates,dim=1)

        if self.template_sizes==None:
          q = self.softmax_weights(self.q0)
        else: 
          q=[]
          for i in range(self.n_templates):
            q.append(self.softmax2(self.q0[i]))

        if self.log:
            x = torch.log(distance_to_template(
                x,
                edge_index,
                self.templates_features,
                self.templates,
                alpha,
                q,
                self.k,
                self.shortest_path))
        else:
            x = distance_to_template(
                x,
                edge_index,
                self.templates_features,
                templates_softmax,
                alpha,
                q,
                self.k,
                self.shortest_path)  
            
             
        return x

        
class LTFGW_semirelaxed(nn.Module):
    """ Layer for the local TFWG 
    
    Computes an embedding for each node by computing a 1 dimensional array of distances between learned templates
      and the node's neighbourhood.

    The distance used is the semi relaxde fused Gromov-Wasserstein distance.
    
    """

    def __init__(
            self,
            n_templates=10,
            n_templates_nodes=10,
            n_features=10,
            k=1,
            mean_init=0,
            std_init=0.001,
            alpha0=None,
            shortest_path=False,
            template_sizes=None,
            reg=0,
            log=False):
        """
        n_templates: int, optional
          Number of graph templates.
        n_templates_nodes: int, optional
          Number of nodes in each template.
        n_features: int, optional
          Number of node features.
        k: int, optional
          Number of hops fot he nodes' neighbourhood.
        mean_init: float, optional
          Mean of the random normal law to initialize the template features.
        std_init:  float, optional
          Std of the random normal law to initialize the template features.
        shortest_path: bool, optional
          If True, the templates are characterized by their shortest path matrix.
          Else, the adjacency matrix is used.
        template_sizes: if None, all template have the same number of nodes. 
          Else, list of the number of nodes of the templates. 
        reg: float
          If None, the distance used is the semi relaxed fused Gromov-Wasserstein.
          Else, the entropic semirelaxed Gromov-Wasserstein is used with the regularisation reg.

        """
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.n_features = n_features
        self.k = k
        self.shortest_path = shortest_path
        self.reg=reg

        templates, templates_features,_ = template_initialisation(
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
        x = semi_relaxed_marginals_to_template(
            x,
            edge_index,
            self.templates_features,
            self.templates,
            alpha,
            self.k,
            self.shortest_path,
            self.device,
            self.reg)
        return x


class LTFGW_no_softmax(nn.Module):
    """ Layer for the local TFWG 

    Computes an embedding for each node by computing a 1 dimensional array of distances between learned templates
      and the node's neighbourhood.

    The distance used is the fused Gromov-Wasserstein distance.

    n_templates: int, optional
        Number of graph templates.
    n_templates_nodes: int, optional
        Number of nodes in each template.
    n_features: int, optional
        Number of node features.
    k: int, optional
        Number of hops fot he nodes' neighbourhood.
    mean_init: float, optional
        Mean of the random normal law to initialize the template features.
    std_init:  float, optional
        Std of the random normal law to initialize the template features.
    train_node_weights: bool, optional
        If True, the node weights are trained.
        Else they are uniform.
    shortest_path: bool, optional
        If True, the templates are characterized by their shortest path matrix.
        Else, the adjacency matrix is used.
    template_sizes: if None, all template have the same number of nodes. 
        Else, list of the number of nodes of the templates. 
    log: bool
       If True the log of the output of the layer is used.
    
    """
    def __init__(
            self,
            n_templates=10,
            n_templates_nodes=10,
            n_features=10,
            k=1,
            alpha0=None,
            mean_init=0,
            std_init=0.001,
            train_node_weights=True,
            shortest_path=False,
            template_sizes=None,
            log=False):
        """ Layer for the local TFWG 

        Computes an embedding for each node by computing a 1 dimensional array of distances between learned templates
        and the node's neighbourhood.

        The distance used is the fused Gromov-Wasserstein distance.

        Parameters
        ----------

        n_templates: int, optional
          Number of graph templates.
        n_templates_nodes: int, optional
          Number of nodes in each template.
        n_features: int, optional
          Number of node features.
        k: int, optional
          Number of hops fot he nodes' neighbourhood.
        mean_init: float, optional
          Mean of the random normal law to initialize the template features.
        std_init:  float, optional
          Std of the random normal law to initialize the template features.
        train_node_weights: bool, optional
          If True, the node weights are trained.
          Else they are uniform.
        shortest_path: bool, optional
          If True, the templates are characterized by their shortest path matrix.
          Else, the adjacency matrix is used.
        template_sizes: if None, all template have the same number of nodes. 
          Else, list of the number of nodes of the templates. 
        log: bool,optional
          If True, the output is the log of the LTFGW layer.
          Else, it is the LTFGW layer.
        """
        super().__init__()

        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.n_features = n_features
        self.k = k
        self.shortest_path = shortest_path
        self.template_sizes=template_sizes
        self.log=log

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
        
        

        self.softmax_weights = nn.Softmax(dim=1)  
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
 #       templates_softmax=torch.softmax(self.templates,dim=1)

        if self.template_sizes==None:
          q = self.softmax_weights(self.q0)
        else: 
          q=[]
          for i in range(self.n_templates):
            q.append(self.softmax2(self.q0[i]))

        if self.log:
            x = torch.log(distance_to_template(
                x,
                edge_index,
                self.templates_features,
                self.templates,
                alpha,
                q,
                self.k,
                self.shortest_path))
        else:
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
    


class TFGWPooling(nn.Module):
    """
    Template Fused Gromov-Wasserstein (TFGW) layer. This layer is a pooling layer for graph neural networks.
        It computes the fused Gromov-Wasserstein distances between the graph and a set of templates.

    Parameters
    ----------
    n_features : int
        Feature dimension of the nodes.
    n_tplt : int
         Number of graph templates.
    n_tplt_nodes : int
        Number of nodes in each template.
    alpha0 : float, optional
        FGW trade-off parameter (0 < alpha < 1). If None alpha is trained, else it is fixed at the given value.
        Weights features (alpha=0) and structure (alpha=1).
    train_node_weights : bool, optional
        If True, the templates node weights are learned.
        Else, they are uniform.
    multi_alpha: bool, optional
        If True, the alpha parameter is a vector of size n_tplt.        
    feature_init_mean: float, optional
        Mean of the random normal law to initialize the template features.
    feature_init_std: float, optional
        Standard deviation of the random normal law to initialize the template features.



    References
    ----------
    .. [52]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Template based graph neural network with optimal transport distances"
    """

    def __init__(self, n_features, n_tplt=2, n_tplt_nodes=2, alpha0=None, train_node_weights=True, multi_alpha=False, feature_init_mean=0., feature_init_std=1.):
        """
        Template Fused Gromov-Wasserstein (TFGW) layer. This layer is a pooling layer for graph neural networks.
            It computes the fused Gromov-Wasserstein distances between the graph and a set of templates.

        Parameters
        ----------
        n_features : int
                Feature dimension of the nodes.
        n_tplt : int
                Number of graph templates.
        n_tplt_nodes : int
                Number of nodes in each template.
        alpha : float, optional
                Trade-off parameter (0 < alpha < 1). If None alpha is trained, else it is fixed at the given value.
                Weights features (alpha=0) and structure (alpha=1).
        train_node_weights : bool, optional
                If True, the templates node weights are learned.
                Else, they are uniform.
        multi_alpha: bool, optional
                If True, the alpha parameter is a vector of size n_tplt.
        feature_init_mean: float, optional
                Mean of the random normal law to initialize the template features.
        feature_init_std: float, optional
                Standard deviation of the random normal law to initialize the template features.

        References
        ----------
        .. [52]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
              "Template based graph neural network with optimal transport distances"

        """
        super().__init__()

        self.n_tplt = n_tplt
        self.n_tplt_nodes = n_tplt_nodes
        self.n_features = n_features
        self.multi_alpha = multi_alpha
        self.feature_init_mean = feature_init_mean
        self.feature_init_std = feature_init_std

        tplt_adjacencies, tplt_features, self.q0 = template_initialisation(self.n_tplt_nodes, self.n_tplt, self.n_features, self.feature_init_mean, self.feature_init_std)
        self.tplt_adjacencies = nn.Parameter(tplt_adjacencies)
        self.tplt_features = nn.Parameter(tplt_features)

        self.softmax = nn.Softmax(dim=1)

        if train_node_weights:
            self.q0 = nn.Parameter(self.q0)

        if alpha0 is None:
            if multi_alpha:
                self.alpha0 =torch.Tensor([0] * self.n_tplt)
            else:
                alpha0 = torch.Tensor([0])
            self.alpha0 = nn.Parameter(alpha0)
        else:
            if multi_alpha:
                self.alpha0 = torch.Tensor([alpha0] * self.n_tplt)
                self.alpha0 = nn.Parameter(self.alpha0) 
            else:
                self.alpha0 = torch.Tensor([alpha0])
            self.alpha0 = torch.logit(alpha0)

    def forward(self, x, edge_index, batch=None):
        alpha = torch.sigmoid(self.alpha0)
        q = self.softmax(self.q0)
        x = distance_to_templates3( edge_index, self.tplt_adjacencies,x,self.tplt_features,q,  alpha,self.multi_alpha, batch)
        return x    