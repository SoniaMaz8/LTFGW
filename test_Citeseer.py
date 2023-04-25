from architectures import GCN_LTFGW,GCN_3_layers
import torch
from tqdm import tqdm
import csv
import datetime
from data.convert_datasets import Citeseer_data

dataset=Citeseer_data()
model=GCN_LTFGW(n_classes=6,N_features=dataset.num_features, N_templates=10,N_templates_nodes=10)
#model=GCN_3_layers(n_classes=6,N_features=dataset.num_features)
model.load_state_dict(torch.load('models/model_Citeseer.pt'))


def test():
      model.eval()
      out = model(dataset.x,dataset.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]  
      test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum()) 
      return test_acc

test_acc=test()
print(test_acc)
