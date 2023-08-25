import numpy as np 
import os.path as ops
node_features = np.loadtxt(ops.join('/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/origin/', 'node_atrri.txt'),dtype=int)
lables = np.loadtxt(ops.join('/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/origin/', 'label.txt'),dtype=int)
edges = np.loadtxt(ops.join('/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/origin/', 'edge_index.txt'),dtype=int)
edges_index = []
edges_x, edges_y = [], []
for e in edges:
    edges_x.append(e[0])
    edges_y.append(e[1])
edges_index.append(edges_x)
edges_index.append(edges_y)
print(edges_index)
interactions = np.loadtxt(ops.join('/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/origin/', 'interactions.txt'),dtype=int)

