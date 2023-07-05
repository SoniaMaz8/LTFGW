import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear, ChebConv, GATConv
from GNN.layers import *
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import APPNP


class GCN(nn.Module):
    def __init__(self, n_classes, n_features, hidden_layer, n_hidden_layer, dropout):
        """
        n_classes: int
           Number of classes for node classification.
        n_features: int
           Number of features for each node.
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_layer = hidden_layer
        self.n_hidden_layers = n_hidden_layer
        self.dropout = dropout

        self.first_conv = GCNConv(self.n_features, self.hidden_layer)
        self.dropout = torch.nn.Dropout(p=self.dropout)

        # list of GCN layers
        self.list_hidden_layer = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.list_hidden_layer.append(
                GCNConv(self.hidden_layer, self.hidden_layer))

        self.last_conv = GCNConv(self.hidden_layer, self.n_classes)

    def forward(self, x, edge_index):

        x = self.first_conv(x, edge_index)
        x = x.relu()
        x = self.dropout(x)

        # go through hidden layers

        for i in range(self.n_hidden_layers):
            x = self.list_hidden_layer[i](x, edge_index)
            x = x.relu()

        x_latent = x
        x = self.last_conv(x, edge_index)
        return x, x_latent


class LTFGW_GCN(nn.Module):
    """"
    Architecture combining a GCN and the LTFGW layer.

        Parameters
        ----------

        n_classes: int
        Number of classes
        n_features: int, optional
            Number of node features.       
        n_templates: int, optional
            Number of graph templates.
        n_templates_nodes: int, optional
            Number of nodes in each template.
        hidden_layer: int
            Hidden dimension.
        dropout: float
            Dropout.
        shortest_path: bool, optional
            If True, the templates are characterized by their shortest path matrix.
            Else, the adjacency matrix is used.       
        k: int, optional
            Number of hops fot he nodes' neighbourhood.
        mean_init: float
            Mean of the random normal law to initialize the template features.
        std_init: float
            Std of the random normal law to initialize the template features. 
        log: bool
            If True the log of the output of the layer is used.  
        alpha0: float
            Trade off parameter for the Fused Gromov-Wasserstein distance.
            If None, it is learned. 
        train_node_weights: bool, optional
            If True, the node weights are trained.
            Else they are uniform.
        template_sizes: if None, all template have the same number of nodes. 
            Else, list of the number of nodes of the templates.      
    """
    def __init__(self, n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log=False,alpha0=None,train_node_weights=True, skip_connection=False ,template_sizes=None):

        """
        Architecture combining a GCN and the LTFGW layer.

        Parameters
        ----------

        n_classes: int
        Number of classes
        n_features: int, optional
            Number of node features.       
        n_templates: int, optional
            Number of graph templates.
        n_templates_nodes: int, optional
            Number of nodes in each template.
        hidden_layer: int
            Hidden dimension.
        dropout: float
            Dropout.
        shortest_path: bool, optional
            If True, the templates are characterized by their shortest path matrix.
            Else, the adjacency matrix is used.       
        k: int, optional
            Number of hops fot he nodes' neighbourhood.
        mean_init: float
            Mean of the random normal law to initialize the template features.
        std_init: float
            Std of the random normal law to initialize the template features. 
        log: bool
            If True the log of the output of the layer is used.  
        alpha0: float
            Trade off parameter for the Fused Gromov-Wasserstein distance.
            If None, it is learned. 
        train_node_weights: bool, optional
            If True, the node weights are trained.
            Else they are uniform.
        template_sizes: if None, all template have the same number of nodes. 
            Else, list of the number of nodes of the templates.     
            
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.hidden_layer = hidden_layer
        self.alpha0 = alpha0
        self.train_node_weights = train_node_weights
        self.skip_connection = skip_connection
        self.drop = dropout
        self.shortest_path = shortest_path
        self.k = k

        self.dropout = torch.nn.Dropout(self.drop)

        self.linear = Linear(self.n_features, self.hidden_layer)
        self.conv1 = GCNConv(self.n_features, self.hidden_layer)
        self.conv2 = GCNConv(
            self.hidden_layer +
            self.n_templates,
            self.n_classes)
        self.conv3 = GCNConv(self.n_templates, self.n_classes)
        self.LTFGW = LTFGW(
            self.n_templates,
            self.n_templates_nodes,
            self.hidden_layer,
            self.k,
            self.alpha0,
            mean_init,
            std_init,
            self.train_node_weights,
            self.shortest_path,
            template_sizes,
            log)

    def forward(self, x, edge_index):

        if self.skip_connection:
            y = self.linear(x)
            y = self.LTFGW(y, edge_index)
            z = self.conv1(x, edge_index)
            z = z.relu()
            x = torch.hstack([z, y])
            x = self.dropout(x)
            x_latent = x
            x = self.conv2(x, edge_index)
        else:
            x = self.LTFGW(x, edge_index)
            x = self.conv3(x, edge_index)
        return x, x_latent


class MLP(nn.Module):
    """
    MLP architecture

    Parameters
    ----------
    n_hidden_layer:int
        Number of linear layers.
    hidden_layer: int
        Hidden dimension.
    dropout: float
        Dropout.
    n_classes: int
        Number of classes for node classification.
    n_features: int
        Number of features for each node.
    """
    def __init__(self, n_hidden_layer, hidden_layer, dropout, n_classes, n_features):
        """
        G architecture

        Parameters
        ----------
        n_hidden_layer:int
           Number of linear layers.
        hidden_layer: int
           Hidden dimension.
        dropout: float
           Dropout.
        n_classes: int
           Number of classes for node classification.
        n_features: int
           Number of features for each node.
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_layers = n_hidden_layer
        self.hidden_layer = hidden_layer
        self.drop = dropout

        self.first_linear = Linear(self.n_features, self.hidden_layer)
        self.dropout1 = torch.nn.Dropout(self.drop)
        self.dropout2 = torch.nn.Dropout(self.drop)

        # list of Linear layers
        self.list_hidden_layer = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.list_hidden_layer.append(
                Linear(self.hidden_layer, self.hidden_layer))

        self.last_linear = Linear(self.hidden_layer, self.n_classes)

    def forward(self, x, edge_index):

        x = self.dropout1(x)
        x = self.first_linear(x)

        x = x.relu()

        # go through hidden layers
        for i in range(self.n_hidden_layers):
            x = self.list_hidden_layer[i](x)
            x = x.relu()

        x_latent = x
        x = self.dropout2(x)
        x = self.last_linear(x)

        return x, x_latent


class LTFGW_MLP(nn.Module):
    """"
    Architecture combining a MLP and the LTFGW layer.

    Parameters
    ----------

    n_classes: int
    Number of classes
    n_features: int, optional
        Number of node features.       
    n_templates: int, optional
        Number of graph templates.
    n_templates_nodes: int, optional
        Number of nodes in each template.
    hidden_layer: int
        Hidden dimension.
    dropout: float
        Dropout.
    shortest_path: bool, optional
        If True, the templates are characterized by their shortest path matrix.
        Else, the adjacency matrix is used.       
    k: int, optional
        Number of hops fot he nodes' neighbourhood.
    mean_init: float
        Mean of the random normal law to initialize the template features.
    std_init: float
        Std of the random normal law to initialize the template features. 
    log: bool
        If True the log of the output of the layer is used.  
    alpha0: float
        Trade off parameter for the Fused Gromov-Wasserstein distance.
        If None, it is learned. 
    train_node_weights: bool, optional
        If True, the node weights are trained.
        Else they are uniform.
    template_sizes: if None, all template have the same number of nodes. 
        Else, list of the number of nodes of the templates.  
    """
    def __init__(self, n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log=False,alpha0=None,train_node_weights=True, skip_connection=False ,template_sizes=None):
        """
        Architecture combining a MLP and the LTFGW layer.

        Parameters
        ----------

        n_classes: int
        Number of classes
        n_features: int, optional
            Number of node features.       
        n_templates: int, optional
            Number of graph templates.
        n_templates_nodes: int, optional
            Number of nodes in each template.
        hidden_layer: int
            Hidden dimension.
        dropout: float
            Dropout.
        shortest_path: bool, optional
            If True, the templates are characterized by their shortest path matrix.
            Else, the adjacency matrix is used.       
        k: int, optional
            Number of hops fot he nodes' neighbourhood.
        mean_init: float
            Mean of the random normal law to initialize the template features.
        std_init: float
            Std of the random normal law to initialize the template features. 
        log: bool
            If True the log of the output of the layer is used.  
        alpha0: float
            Trade off parameter for the Fused Gromov-Wasserstein distance.
            If None, it is learned. 
        train_node_weights: bool, optional
            If True, the node weights are trained.
            Else they are uniform.
        template_sizes: if None, all template have the same number of nodes. 
            Else, list of the number of nodes of the templates.  

        """

        super().__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.hidden_layer = hidden_layer
        self.alpha0 = alpha0
        self.train_node_weights = train_node_weights
        self.skip_connection = skip_connection
        self.drop = dropout
        self.shortest_path = shortest_path
        self.k = k
        self.log=log

        self.dropout2 = torch.nn.Dropout(self.drop)

        self.Linear1 = Linear(self.n_features, self.hidden_layer)
        self.Linear2 = Linear(
            self.hidden_layer +
            self.n_templates,
            self.n_classes)
        self.Linear3 = Linear(self.n_templates, self.n_classes)
        self.LTFGW = LTFGW(
            self.n_templates,
            self.n_templates_nodes,
            self.hidden_layer,
            self.k,
            self.alpha0,
            mean_init,
            std_init,
            self.train_node_weights,
            self.shortest_path,
            template_sizes,
            self.log)

    def forward(self, x, edge_index):

        x = self.Linear1(x)

        if self.skip_connection:
            y = self.LTFGW(x, edge_index)
            x = torch.hstack([x, y])
            x = x.relu()
            x = self.dropout2(x)
            x = self.Linear2(x)
            x_latent = x

        else:
            x = self.LTFGW(x, edge_index)
            x = self.Linear3(x)

        return x, x_latent


class ChebNet(torch.nn.Module):
    def __init__(self, dropout, n_classes, n_features):
        super(ChebNet, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.drop = dropout

        self.conv1 = ChebConv(self.n_features, 32, K=2)
        self.conv2 = ChebConv(32, self.n_classes, K=2)
        self.dropout = self.drop

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_latent = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), x_latent


class GAT(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN_JK(torch.nn.Module):
    def __init__(self, n_classes, n_features, hidden_layer):
        in_channels = n_features
        out_channels = n_classes

        self.drop = dropout

        super(GCN_JK, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_layer)
        self.conv2 = GCNConv(
            hidden_layer,
            hidden_layer)
        self.lin1 = torch.nn.Linear(2 * hidden_layer, out_channels)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='cat',
                                   channels=hidden_layer,
                                   num_layers=8
                                   )

    def forward(self, x, edge_index):

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.drop, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=self.drop, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return x, x


class LTFGW_MLP_semirelaxed(nn.Module):
    """"
    Architecture combining a MLP and the LTFGW_semirelaxed layer.

    Parameters
    ----------

    n_classes: int
        Number of classes
    n_features: int, optional
        Number of node features.       
    n_templates: int, optional
        Number of graph templates.
    n_templates_nodes: int, optional
        Number of nodes in each template.
    hidden_layer: int
        Hidden dimension.
    dropout: float
        Dropout.
    shortest_path: bool, optional
        If True, the templates are characterized by their shortest path matrix.
        Else, the adjacency matrix is used.       
    k: int, optional
        Number of hops fot he nodes' neighbourhood.
    mean_init: float
        Mean of the random normal law to initialize the template features.
    std_init: float
        Std of the random normal law to initialize the template features. 
    log: bool
        If True the log of the output of the layer is used.  
    alpha0: float
        Trade off parameter for the Fused Gromov-Wasserstein distance.
        If None, it is learned. 
    train_node_weights: bool, optional
        If True, the node weights are trained.
        Else they are uniform.
    template_sizes: if None, all template have the same number of nodes. 
        Else, list of the number of nodes of the templates.   
    reg: float
        Regularisation parameter for the semi-relaxed Fused Gromov Wasserstein distance.     
                   
    """
    def __init__(self, n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log=False,alpha0=None,train_node_weights=True, skip_connection=True ,template_sizes=None, reg=0):
        """"
        Architecture combining a MLP and the LTFGW_semirelaxed layer.

        Parameters
        ----------

        n_classes: int
            Number of classes
        n_features: int, optional
            Number of node features.       
        n_templates: int, optional
            Number of graph templates.
        n_templates_nodes: int, optional
            Number of nodes in each template.
        hidden_layer: int
            Hidden dimension.
        dropout: float
            Dropout.
        shortest_path: bool, optional
            If True, the templates are characterized by their shortest path matrix.
            Else, the adjacency matrix is used.       
        k: int, optional
            Number of hops fot he nodes' neighbourhood.
        mean_init: float
            Mean of the random normal law to initialize the template features.
        std_init: float
            Std of the random normal law to initialize the template features. 
        log: bool
            If True the log of the output of the layer is used.  
        alpha0: float
            Trade off parameter for the Fused Gromov-Wasserstein distance.
            If None, it is learned. 
        train_node_weights: bool, optional
            If True, the node weights are trained.
            Else they are uniform.
        template_sizes: if None, all template have the same number of nodes. 
            Else, list of the number of nodes of the templates. 
        reg: float
            Regularisation parameter for the semi-relaxed Fused Gromov Wasserstein distance.     
        
        """

        super().__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.hidden_layer = hidden_layer
        self.alpha0 = alpha0
        self.train_node_weights = train_node_weights
        self.skip_connection = skip_connection
        self.drop = dropout
        self.shortest_path = shortest_path
        self.k = k
        self.reg=reg
        self.log=log

        self.dropout1 = torch.nn.Dropout(self.drop)
        self.dropout2 = torch.nn.Dropout(self.drop)

        self.Linear1 = Linear(self.n_features, self.hidden_layer)
        self.Linear2 = Linear(
            self.hidden_layer +
            self.n_templates*self.n_templates_nodes,
            self.n_classes)
        self.Linear3 = Linear(self.n_templates, self.n_classes)
        self.LTFGW = LTFGW_semirelaxed(
            self.n_templates,
            self.n_templates_nodes,
            self.hidden_layer,
            self.k,
            mean_init,
            std_init,
            self.alpha0,
            self.shortest_path,
            template_sizes,
            self.reg,
            self.log)

    def forward(self, x, edge_index):

        x = self.dropout1(x)
        x = self.Linear1(x)

        if self.skip_connection:
            y = self.LTFGW(x, edge_index)
            x = torch.hstack([x, y])
            x = x.relu()
            x_latent = x
            x = self.dropout2(x)
            x = self.Linear2(x)
            

        else:
            x = self.LTFGW(x, edge_index)
            x = self.Linear3(x)

        return x, x_latent


class LTFGW_MLP_dropout(nn.Module):
    """"
    Architecture combining a MLP and the LTFGW layer. One dropout is added at the begining.

    Parameters
    ----------

    n_classes: int
        Number of classes
    n_features: int, optional
        Number of node features.       
    n_templates: int, optional
        Number of graph templates.
    n_templates_nodes: int, optional
        Number of nodes in each template.
    hidden_layer: int
        Hidden dimension.
    dropout: float
        Dropout.
    shortest_path: bool, optional
        If True, the templates are characterized by their shortest path matrix.
        Else, the adjacency matrix is used.       
    k: int, optional
        Number of hops fot he nodes' neighbourhood.
    mean_init: float
        Mean of the random normal law to initialize the template features.
    std_init: float
        Std of the random normal law to initialize the template features. 
    log: bool
        If True the log of the output of the layer is used.  
    alpha0: float
        Trade off parameter for the Fused Gromov-Wasserstein distance.
        If None, it is learned. 
    train_node_weights: bool, optional
        If True, the node weights are trained.
        Else they are uniform.
    template_sizes: if None, all template have the same number of nodes. 
        Else, list of the number of nodes of the templates. 
    """
    def __init__(self, n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log=False,alpha0=None,train_node_weights=True, skip_connection=False ,template_sizes=None):
        """
        Architecture combining a MLP and the LTFGW layer. One dropout is added at the begining.

        Parameters
        ----------

        n_classes: int
            Number of classes
        n_features: int, optional
            Number of node features.       
        n_templates: int, optional
            Number of graph templates.
        n_templates_nodes: int, optional
            Number of nodes in each template.
        hidden_layer: int
            Hidden dimension.
        dropout: float
            Dropout.
        shortest_path: bool, optional
            If True, the templates are characterized by their shortest path matrix.
            Else, the adjacency matrix is used.       
        k: int, optional
            Number of hops fot he nodes' neighbourhood.
        mean_init: float
            Mean of the random normal law to initialize the template features.
        std_init: float
            Std of the random normal law to initialize the template features. 
        log: bool
            If True the log of the output of the layer is used.  
        alpha0: float
            Trade off parameter for the Fused Gromov-Wasserstein distance.
            If None, it is learned. 
        train_node_weights: bool, optional
            If True, the node weights are trained.
            Else they are uniform.
        template_sizes: if None, all template have the same number of nodes. 
            Else, list of the number of nodes of the templates. 

        """

        super().__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.hidden_layer = hidden_layer
        self.alpha0 = alpha0
        self.train_node_weights = train_node_weights
        self.skip_connection = skip_connection
        self.drop = dropout
        self.shortest_path = shortest_path
        self.k = k
        self.template_sizes=template_sizes
        self.log=log

        self.dropout1 = torch.nn.Dropout(self.drop)
        self.dropout2 = torch.nn.Dropout(self.drop)

        self.Linear1 = Linear(self.n_features, self.hidden_layer)
        self.Linear2 = Linear(
            self.hidden_layer +
            self.n_templates,
            self.n_classes)
        self.Linear3 = Linear(self.n_templates, self.n_classes)

        self.LTFGW = LTFGW(
            self.n_templates,
            self.n_templates_nodes,
            self.hidden_layer,
            self.k,
            self.alpha0,
            mean_init,
            std_init,
            self.train_node_weights,
            self.shortest_path,
            self.template_sizes,
            self.log)

    def forward(self, x, edge_index):

        x = self.dropout1(x)
        x = self.Linear1(x)
        
        if self.skip_connection:
            y = self.LTFGW(x, edge_index)
            x = torch.hstack([x, y])
            x = x.relu()
            x = self.dropout2(x)
            x = self.Linear2(x)
            x_latent = x

        else:
            x = self.LTFGW(x, edge_index)
            x = self.Linear3(x)
        
        return x, x_latent
    

class LTFGW_GCN_dropout(nn.Module):
    """"
    Architecture combining a MLP and the LTFGW layer. One dropout is added at the begining.

    Parameters
    ----------

    n_classes: int
        Number of classes
    n_features: int, optional
        Number of node features.       
    n_templates: int, optional
        Number of graph templates.
    n_templates_nodes: int, optional
        Number of nodes in each template.
    hidden_layer: int
        Hidden dimension.
    dropout: float
        Dropout.
    shortest_path: bool, optional
        If True, the templates are characterized by their shortest path matrix.
        Else, the adjacency matrix is used.       
    k: int, optional
        Number of hops fot he nodes' neighbourhood.
    mean_init: float
        Mean of the random normal law to initialize the template features.
    std_init: float
        Std of the random normal law to initialize the template features. 
    log: bool
        If True the log of the output of the layer is used.  
    alpha0: float
        Trade off parameter for the Fused Gromov-Wasserstein distance.
        If None, it is learned. 
    train_node_weights: bool, optional
        If True, the node weights are trained.
        Else they are uniform.
    template_sizes: if None, all template have the same number of nodes. 
        Else, list of the number of nodes of the templates. 
    """
    def __init__(self, n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log=False,alpha0=None,train_node_weights=True, skip_connection=False ,template_sizes=None):
        """
        Architecture combining a MLP and the LTFGW layer. One dropout is added at the begining.

        Parameters
        ----------

        n_classes: int
            Number of classes
        n_features: int, optional
            Number of node features.       
        n_templates: int, optional
            Number of graph templates.
        n_templates_nodes: int, optional
            Number of nodes in each template.
        hidden_layer: int
            Hidden dimension.
        dropout: float
            Dropout.
        shortest_path: bool, optional
            If True, the templates are characterized by their shortest path matrix.
            Else, the adjacency matrix is used.       
        k: int, optional
            Number of hops fot he nodes' neighbourhood.
        mean_init: float
            Mean of the random normal law to initialize the template features.
        std_init: float
            Std of the random normal law to initialize the template features. 
        log: bool
            If True the log of the output of the layer is used.  
        alpha0: float
            Trade off parameter for the Fused Gromov-Wasserstein distance.
            If None, it is learned. 
        train_node_weights: bool, optional
            If True, the node weights are trained.
            Else they are uniform.
        template_sizes: if None, all template have the same number of nodes. 
            Else, list of the number of nodes of the templates. 

        """

        super().__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.hidden_layer = hidden_layer
        self.alpha0 = alpha0
        self.train_node_weights = train_node_weights
        self.skip_connection = skip_connection
        self.drop = dropout
        self.shortest_path = shortest_path
        self.k = k
        self.template_sizes=template_sizes
        self.log=log

        self.dropout1 = torch.nn.Dropout(self.drop)
        self.dropout2 = torch.nn.Dropout(self.drop)

        self.first_conv = GCNConv(self.n_features, self.hidden_layer)
        self.conv = GCNConv(self.hidden_layer, self.n_classes)


        self.LTFGW = LTFGW(
            self.n_templates,
            self.n_templates_nodes,
            self.hidden_layer,
            self.k,
            self.alpha0,
            mean_init,
            std_init,
            self.train_node_weights,
            self.shortest_path,
            self.template_sizes,
            self.log)

    def forward(self, x, edge_index):

        x = self.dropout1(x)
        x = self.first_conv(x,edge_index)
        
        if self.skip_connection:
            y = self.LTFGW(x, edge_index)
            x = torch.hstack([x, y])
            x = x.relu()
            x = self.dropout2(x)
            x = self.conv(x,edge_index)
            x_latent = x

        else:
            x = self.LTFGW(x, edge_index)
            x = self.Linear3(x)
        
        return x, x_latent    


class LTFGW_MLP_dropout_relu(nn.Module):
    """"
    Architecture combining a MLP and the LTFGW layer. The relu is placed right after the first Linear.

    Parameters
    ----------

    n_classes: int
        Number of classes
    n_features: int, optional
        Number of node features.       
    n_templates: int, optional
        Number of graph templates.
    n_templates_nodes: int, optional
        Number of nodes in each template.
    hidden_layer: int
        Hidden dimension.
    dropout: float
        Dropout.
    shortest_path: bool, optional
        If True, the templates are characterized by their shortest path matrix.
        Else, the adjacency matrix is used.       
    k: int, optional
        Number of hops fot he nodes' neighbourhood.
    mean_init: float
        Mean of the random normal law to initialize the template features.
    std_init: float
        Std of the random normal law to initialize the template features. 
    log: bool
        If True the log of the output of the layer is used.  
    alpha0: float
        Trade off parameter for the Fused Gromov-Wasserstein distance.
        If None, it is learned. 
    train_node_weights: bool, optional
        If True, the node weights are trained.
        Else they are uniform.
    template_sizes: if None, all template have the same number of nodes. 
        Else, list of the number of nodes of the templates. 

    """
    def __init__(self, n_classes, n_features ,n_templates,n_templates_nodes,hidden_layer,dropout,shortest_path,k,mean_init,std_init,log=False,alpha0=None,train_node_weights=True, skip_connection=False ,template_sizes=None):
        """
        Architecture combining a MLP and the LTFGW layer. The relu is placed right after the first Linear.

        Parameters
        ----------

        n_classes: int
            Number of classes
        n_features: int, optional
            Number of node features.       
        n_templates: int, optional
            Number of graph templates.
        n_templates_nodes: int, optional
            Number of nodes in each template.
        hidden_layer: int
            Hidden dimension.
        dropout: float
            Dropout.
        shortest_path: bool, optional
            If True, the templates are characterized by their shortest path matrix.
            Else, the adjacency matrix is used.       
        k: int, optional
            Number of hops fot he nodes' neighbourhood.
        mean_init: float
            Mean of the random normal law to initialize the template features.
        std_init: float
            Std of the random normal law to initialize the template features. 
        log: bool
            If True the log of the output of the layer is used.  
        alpha0: float
            Trade off parameter for the Fused Gromov-Wasserstein distance.
            If None, it is learned. 
        train_node_weights: bool, optional
            If True, the node weights are trained.
            Else they are uniform.
        template_sizes: if None, all template have the same number of nodes. 
            Else, list of the number of nodes of the templates. 


        """

        super().__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.n_templates = n_templates
        self.n_templates_nodes = n_templates_nodes
        self.hidden_layer = hidden_layer
        self.alpha0 = alpha0
        self.train_node_weights = train_node_weights
        self.skip_connection = skip_connection
        self.drop = dropout
        self.shortest_path = shortest_path
        self.k = k

        self.dropout1 = torch.nn.Dropout(self.drop)
        self.dropout2 = torch.nn.Dropout(self.drop)

        self.Linear1 = Linear(self.n_features, self.hidden_layer)
        self.Linear2 = Linear(
            self.hidden_layer +
            self.n_templates,
            self.n_classes)
        self.Linear3 = Linear(self.n_templates, self.n_classes)
        self.LTFGW = LTFGW(
            self.n_templates,
            self.n_templates_nodes,
            self.hidden_layer,
            self.k,
            self.alpha0,
            mean_init,
            std_init,
            self.train_node_weights,
            self.shortest_path,
            template_sizes)

    def forward(self, x, edge_index):

        x = self.dropout1(x)
        x = self.Linear1(x)
        x = x.relu()
        
        if self.skip_connection:
            y = self.LTFGW(x, edge_index)
            x = torch.hstack([x, y])
            x = self.dropout2(x)
            x = self.Linear2(x)
            x_latent = x

        else:
            x = self.LTFGW(x, edge_index)
            x = self.Linear3(x)

        return x, x_latent