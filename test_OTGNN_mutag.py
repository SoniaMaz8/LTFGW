from OT_GNN_layer import OT_GNN_layer
import torch

dataset_all=torch.load('test_mutag.pt')
model=OT_GNN_layer()
model.load_state_dict(torch.load('checkpoints/model_mutag.pt')['model_state_dict'])



def test():
    total_test_acc=0 
    for i in range(len(dataset_all)):
      dataset=dataset_all[i]
      model.eval()
      out = model(dataset.x,dataset.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred == dataset.x.argmax(dim=1)
      test_acc = int(test_correct.sum())/dataset.num_nodes
      total_test_acc+=test_acc
    return total_test_acc/len(dataset_all)


test_acc=test()
print(test_acc)
