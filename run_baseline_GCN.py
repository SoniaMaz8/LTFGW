import torch
from torch_geometric.nn import GCNConv, GATConv,CuGraphSAGEConv, TAGConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

#%% Dataset

#dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
#data = dataset[0] 

data = torch.load('toy_graph2.pt')
dataset=torch.load('toy_graph2.pt')

print(data.train_mask)

#%% GCN Model

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

torch.manual_seed(0)
model = GCN(hidden_channels=16)
print(model)

#%% Train the model



optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      out = model(data.x, data.edge_index) 
      print(out)
      pred = out.argmax(dim=1) 
      loss = criterion(out[data.train_mask], data.y[data.train_mask]) 
      loss.backward()  
      optimizer.step()  
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1) 
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc


test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
