from OT_GNN_layer import OT_GNN_layer
import torch

dataset=torch.load('toy_graph1.pt')
model=OT_GNN_layer()

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


def train():
      model.train()
      optimizer.zero_grad()  
      out = model(dataset.x,dataset.edge_index) 
      pred = out.argmax(dim=1)         # Use the class with highest probability
      train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]  
      train_acc = int(train_correct.sum()) / int(dataset.train_mask.sum())  
      loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  
      loss.backward()  
      optimizer.step()  
      return loss, train_acc


Loss=[]
Train_accuracy=[]
for epoch in range(50):
     loss,train_acc = train()
     Train_accuracy.append(train_acc) 
     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},Train Accuracy: {train_acc:.4f}')
     Loss.append(loss.detach().numpy())

   
