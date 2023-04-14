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



#torch.save({
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            }, 'model_Citeseer.pt')

#print(Train_accuracy)
#print(Loss)

#np.save(Loss, 'Loss.npy')

#model.load_state_dict(torch.load('model2.pt')['model_state_dict'])

#test_acc = test(dataset)
#print(f'Test Accuracy 1: {test_acc:.4f}')  

#test_acc = test(dataset2)
#print(f'Test Accuracy 2: {test_acc:.4f}')  

#model = OT_GNN_layer()
#model.load_state_dict(torch.load('model1'))
#model.eval()


#for name, param in model.named_parameters():
#    print(name)
#    print(param)     