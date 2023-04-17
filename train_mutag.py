from OT_GNN_layer import OT_GNN_layer
import torch
import time
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np

dataset=torch.load('data/train_mutag.pt')
model=OT_GNN_layer()

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

batch_size=2

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train():
    model.train()
    total_loss = 0
    total_train_acc=0
    for idx,data in enumerate(train_loader):
        print(idx)
        optimizer.zero_grad()
        out = model(data.x,data.edge_index) 
        pred = out.argmax(dim=1) 
        train_correct = pred == data.x.argmax(dim=1)  #mutag node labels are one hot encoding in x
        train_acc = int(train_correct.sum())/data.num_nodes
        total_train_acc+=train_acc
        loss = criterion(out,data.x.argmax(dim=1))
        total_loss += loss.item() * data.num_graphs  #data.num_graphs=batchsize
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset), total_train_acc / len(train_loader)


for epoch in range(1, 10):
    loss,train_acc = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train accuracy: {train_acc:.4f}')


