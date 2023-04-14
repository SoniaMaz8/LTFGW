import torch
from torch_geometric.nn import GINConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

data = torch.load('toy_graph1.pt')
dataset=torch.load('toy_graph1.pt')

#%% GIN Model

class GCN_GIN(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GINConv(Sequential(Linear(dataset.num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin = Linear(dim_h, dataset.num_classes)
 
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.lin(x)
        return x

model = GCN_GIN(dim_h=16)
print(model)

#%% Train the model

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  
      out = model(data.x, data.edge_index)  
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


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
