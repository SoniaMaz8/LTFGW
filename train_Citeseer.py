from torch_geometric.loader import NeighborLoader
from torch_geometric.data import NeighborSampler
from OT_GNN_layer_Citeseer import OT_GNN_layer
import torch
import time
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
from torch.optim.lr_scheduler import StepLR

dataset=torch.load('data/graph_Citeseer.pt')
torch.manual_seed(123456)

train_loader = NeighborLoader(dataset,num_neighbors= [-1],
    batch_size=4,
    input_nodes=dataset.train_mask,shuffle=True)


model=OT_GNN_layer(n_classes=6,N_features=dataset.num_features, N_templates=10,N_templates_nodes=10)

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train_epoch():
    model.train()
    total_loss = 0
    total_train_acc=0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x,data.edge_index) 
        pred = out.argmax(dim=1) 
        train_correct = pred[data.train_mask] == data.y[data.train_mask]   
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
        total_train_acc+=train_acc
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        total_loss += loss.item() 
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader), total_train_acc / len(train_loader)


Loss=[]
Train_acc=[]
for epoch in range(1, 50):
    loss,train_acc = train_epoch()
    Loss.append(loss)
    Train_acc.append(train_acc)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train accuracy: {train_acc:.4f}')

np.save('Loss.npy',Loss)  
np.save('Train_acc.npy',Train_acc) 

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'model_Citeseer.pt')

