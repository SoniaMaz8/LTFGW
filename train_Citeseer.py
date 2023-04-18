from torch_geometric.loader import NeighborLoader
from torch_geometric.data import NeighborSampler
from OT_GNN_layer import OT_GNN_layer
import torch
import time
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

dataset=torch.load('Citeseer_data/graph_Citeseer.pt')


train_loader = NeighborLoader(dataset,num_neighbors= [-1],
    batch_size=30,
    input_nodes=dataset.train_mask)

model=OT_GNN_layer()

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

Loss=[]
def train():
    model.train()
    total_loss = 0
    total_train_acc=0
    for idx,data in enumerate(train_loader):
        print(idx)
        optimizer.zero_grad()
        out = model(data.x,data.edge_index) 
        pred = out.argmax(dim=1) 
        train_correct = pred[data.train_mask] == data.y[data.train_mask]   
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
        total_train_acc+=train_acc
        print(train_acc)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        Loss.append(loss.detach().numpy())
        total_loss += loss.item() * len(pred)  #data.num_graphs=batchsize
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset), total_train_acc / len(train_loader)

for epoch in range(1, 10):
    loss,train_acc = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train accuracy: {train_acc:.4f}')