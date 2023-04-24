import torch
import torch.nn as nn
from utils import distance_to_template
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv, Linear
import torch.nn.functional as F


def construct_templates(data,N_nodes=10,N_templates=10):
    """"
    This function returns templates as subgraphs of the graph. The subgraphs are neighbourhoods of high degree nodes.
    Input:
        g: graph as input
        N_nodes: number of nodes in each template
        N_templates: number of templates
    Output:
        Templates: list of the adjacency matrices of the templates
        Templates_features: list of the node features of the templates
    """
    model=GCN(len(data.x[0]))
    model.eval()
    out=model(data.x,data.edge_index)
    n=len(data.x)
    adjacency=graph_to_adjacency(len(data.x),data.edge_index)
    indices=torch.randint(0,n,[N_nodes,N_templates])
    Templates=[]
    Templates_features=[]
    for i in range(N_templates):
       Templates.append(adjacency[indices[i],:][:,indices[i]])
       Templates_features.append(out[indices[i]])
    Templates_features=torch.stack(Templates_features) 
    Templates=torch.stack(Templates)
    return Templates,Templates_features


#More complicated initialisation that ensures some connectivity in the templates.
def construct_templates2(data,N_nodes=10,N_templates=10):
    """"
    This function returns templates as subgraphs of the graph. The subgraphs are neighbourhoods of high degree nodes.
    Input:
        g: graph as input
        N_nodes: number of nodes in each template
        N_templates: number of templates
    Output:
        Templates: list of the adjacency matrices of the templates
        Templates_features: list of the node features of the templates
    """
    model=GCN(len(data.x[0]))
    model.eval()
    out=model(data.x,data.edge_index)
    g=GraphData(out,data.edge_index)
    deg = degree(g.edge_index[0], g.num_nodes)
    high_degree_nodes=torch.where(deg>10)
    indexes=torch.randint(len(high_degree_nodes[0]),[N_templates])  #high degree nodes selection (degree>10)
    indexes=high_degree_nodes[0][indexes]     
    Templates=[]
    Templates_features=[]
    for idx in indexes:
        temp=k_hop_subgraph(idx.item(),2,edge_index=g.edge_index,relabel_nodes=False)  #neighbourhood  for the graph
        deg = degree(temp[1][0], torch.max(temp[0])+1)
        temp_high_degree_nodes=torch.where(deg>0)   #we select the subgraph with nodes of degree >0
        v=temp_high_degree_nodes[0]
        if len(v)<N_nodes:
          raise ValueError('not enough nodes in the template')
        temp_high_degree_nodes=v[torch.randperm(len(v))][:N_nodes]  #random node selection
        temp_graph=subgraph_pyg(temp_high_degree_nodes,temp[1])
        u=temp_graph[0]
        _,temp_sorted = torch.sort(u)
        temp_graph=graph_to_adjacency(N_nodes,temp_sorted)
        temp_graph=temp_graph.double()
        Templates.append(temp_graph)
        Templates_features.append(g.x[temp_high_degree_nodes])
    Templates_features=torch.stack(Templates_features) 
    Templates=torch.stack(Templates)
    return Templates,Templates_features


class LTFGW(nn.Module):
    """ Layer for the local TFGW """
    def __init__(self, N_templates=10,N_templates_nodes=10,alpha_is_param=False):
        """
        N_templates: number of graph templates
        N_templates_nodes: number of nodes in each template
        """
        super().__init__()

        self.N_templates= N_templates
        self.N_templates_nodes=N_templates_nodes
        
        #templates initilisation as subgraphs of the dataset after one GCN layer

        templates,templates_features=construct_templates(dataset,self.N_templates_nodes,self.N_templates)
        templates=templates.type(torch.FloatTensor)
        self.templates=nn.Parameter(templates)
        self.templates_features = nn.Parameter(templates_features)

    def forward(self, x, edge_index):
        x=distance_to_template(x,edge_index,self.templates_features,self.templates)
        return x
        

class OT_GNN_layer(nn.Module):
    def __init__(self,n_classes=2, N_templates=10,N_templates_nodes=10,hidden_layer=20):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_templates=N_templates
        self.N_templates_nodes=N_templates_nodes
        self.hidden_layer=hidden_layer
        
        self.conv1=GCNConv(self.N_features,self.hidden_layer)
        self.conv2=GCNConv(self.hidden_layer,self.hidden_layer)   
        self.LTFGW=LTFGW(self.N_templates,self.N_templates_nodes)
        self.linear=Linear(self.N_templates+self.hidden_layer, self.n_classes)

    def forward(self, x, edge_index):
        x=self.conv1(x, edge_index)
        x = x.relu() 
        x=self.conv2(x, edge_index)
        y=self.LTFGW(x,edge_index)
        x=torch.hstack([x,y])
        x=self.linear(x)
        x = x.relu()  
        return  x

#LTFWG replaced by a GCN, for comparison 

class GCN_layer(nn.Module):
    def __init__(self,n_classes=2,N_features=10,hidden_layer=20):
        """
        n_classes: number of classes for node classification
        """
        super().__init__()
    
        self.n_classes=n_classes
        self.N_features=N_features
        self.hidden_layer=hidden_layer
        
        self.conv1=GCNConv(self.N_features,self.hidden_layer)
        self.conv2=GCNConv(self.hidden_layer,self.hidden_layer)
        self.conv3=GCNConv(self.hidden_layer,self.n_classes)  

    def forward(self, x, edge_index):
        x=self.conv1(x, edge_index)
        x = x.relu() 
        x=self.conv2(x, edge_index)
        x = x.relu() 
        x=self.conv3(x, edge_index)
        x = x.relu()  
        return  x        
