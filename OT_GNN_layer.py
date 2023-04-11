import torch
import torch.nn as nn
from utils import distance_to_template,graph_to_adjacency, adjacency_to_graph
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv, Linear

dataset=torch.load('toy_graph1.pt')

class OT_GNN_layer(nn.Module):
    def __init__(self,N_templates=10,N_templates_nodes=10):
        """
        N_templates: number of templates
        N_templates_nodes: number of nodes in each template
        """
        super().__init__()
        self.N_templates=N_templates
        self.N_templates_nodes=N_templates_nodes

        templates=torch.Tensor(N_templates,N_templates_nodes,N_templates_nodes)  #templates adjacency matrices 
        self.templates = nn.Parameter(templates)

        templates_features=torch.Tensor(N_templates,N_templates_nodes,dataset.num_features)
        self.templates_features = nn.Parameter(templates_features)

        self.conv1= GCNConv(dataset.num_features, hidden_channels)
        self.conv2= GCNConv(hidden_channels, dataset.num_features)
        self.linear=Linear(self.N_templates, dataset.num_classes)

        self.C=graph_to_adjacency(dataset.num_nodes,dataset.edge_index)

        # initialize adjacency matrices for the templates
        nn.init.uniform_(self.templates)
        nn.init.uniform_(self.templates_features)

    def forward(self, G):
        x=self.conv1(G.x,G.edge_index)
        x = x.relu()
        x=self.conv2(x,G.edge_index)
        x=distance_to_template(self.C,x,self.templates,self.templates_features,1,G.num_features)
        x=self.linear(x)
        return  x
    

torch.manual_seed(0)
model=OT_GNN_layer()

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
      model.train()
      optimizer.zero_grad()  
      out = model(dataset)  
      loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask]) 
      loss.backward()  
      optimizer.step() 
      return loss

def test():
      model.eval()
      out = model(dataset)
      print(out)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      print(pred)
      test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]  
      test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum())  
      return test_acc

for epoch in range(1, 10):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')   


