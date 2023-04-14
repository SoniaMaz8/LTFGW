from utils import subgraph,adjacency_to_graph,visualize_graph, graph_to_adjacency, distance_to_template
import numpy as np
from sklearn.cluster import KMeans
import dgl 
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from torch_geometric.datasets import Entities
from scipy.sparse import coo_matrix,coo_array
import torch
import time
import ot



M=torch.tensor([[1.0265, 0.7005, 0.7430, 0.7471, 1.3834],
        [1.0265, 0.7005, 0.7430, 0.7471, 1.3834],
        [1.0265, 0.7005, 0.7430, 0.7471, 1.3834],
        [1.8092, 0.6406, 0.6534, 1.1416, 1.2153]])

C1=torch.tensor([[0., 1., 1., 1.],
        [1., 0., 1., 0.],
        [1., 1., 0., 0.],
        [1., 0., 0., 0.]])

C2=torch.tensor([[0.6156, 0.4567, 0.2380, 0.2496, 0.2525],
        [0.5520, 0.5763, 0.4519, 1.0517, 0.3519],
        [0.3075, 0.0935, 0.4523, 0.3647, 0.3835],
        [0.6192, 0.1701, 0.9379, 0.4755, 0.3896],
        [0.4057, 0.5402, 0.1720, 0.6583, 0.7465]])

p=torch.ones(4)/4
q=torch.ones(5)/5


start=time.time()
dist=ot.gromov.fused_gromov_wasserstein2(M, C1, C2, p, q,alpha=0.5,symmetric=True)
end=time.time()
print(end-start)

start=time.time()
dist=ot.gromov.fused_gromov_wasserstein2(M.T, C2, C1, q, p,alpha=0.5,symmetric=True)
end=time.time()
print(end-start)