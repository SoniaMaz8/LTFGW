from architectures import GCN_LTFGW
import torch
from tqdm import tqdm
import csv
import datetime

dataset=torch.load('data/toy_graph1.pt')
model=GCN_LTFGW(n_classes=3,N_features=3)


def test():
      model.eval()
      out = model(dataset.x,dataset.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]  
      test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum()) 
      return test_acc

test_acc=test()
print(test_acc)




