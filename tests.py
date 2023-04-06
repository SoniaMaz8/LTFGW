from utils import subgraph,adjacency_to_graph,visualize_graph
import numpy as np
from sklearn.cluster import KMeans
import dgl 
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment

x=np.array([0,1,2,3,4])

C=np.array([[0,1,1,0,0],
   [1,0,1,0,0],
   [1,1,0,1,0],
   [0,0,1,0,1],
   [0,0,0,1,0]])

G=adjacency_to_graph(C,x)

#visualize_graph(G)

#C_sub,x_sub=subgraph(C,x,3,1)

#G_sub=adjacency_to_graph(C_sub,x_sub)

#visualize_graph(G_sub)


X=np.load('distance2.npy')
print(X.shape)
kmeans = KMeans(n_clusters=6, random_state=0, n_init="auto").fit(X)



dataset=dgl.data.CiteseerGraphDataset()
g = dataset[0]
label = g.ndata['label']


print(metrics.rand_score(label, kmeans.labels_))
print(metrics.adjusted_mutual_info_score(label, kmeans.labels_))
print(metrics.homogeneity_score(label, kmeans.labels_))
print(metrics.completeness_score(label, kmeans.labels_))

colors=['red','blue','green','orange','black','purple']

#plt.figure(1)
#for i in range(len(label)):
#    if i%20==0:
#      plt.scatter(X[i,4],X[i,5],color=colors[label[i]])
#plt.show()    

cm=metrics.confusion_matrix(label,kmeans.labels_)

#plt.figure(1)
#plt.imshow(cm)
#plt.show()


def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

indexes = linear_assignment(_make_cost_m(cm))
js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
cm2 = cm[:, js]

#plt.figure(2)
#plt.imshow(cm2)
#plt.show()