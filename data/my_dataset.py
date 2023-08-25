import os.path as osp
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

class CustomCora(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        super(CustomCora, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['node_atrri.txt', 'label.txt', 'edge_index.txt','interactions.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Downloading is not required for your case as you already have the data.
        pass

    def process(self):
        node_features = np.loadtxt(osp.join('/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/raw/', 'node_atrri.txt'),dtype=int)
        lables = np.loadtxt(osp.join('/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/raw/', 'label.txt'),dtype=int)
        edges = np.loadtxt(osp.join('/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/raw/', 'edge_index.txt'),dtype=int)
        interactions = np.loadtxt(osp.join('/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/raw/', 'interactions.txt'),dtype=int)
     
        node = node_features


        
        edges_index = []
        edges_x, edges_y = [], []
        for e in edges:
            edges_x.append(e[0])
            edges_y.append(e[1])
        edges_index.append(edges_x)
        edges_index.append(edges_y)
        
        edge_attrs = interactions

        
        

        x = range(node_features.shape[0])
        
        train_size = 0.8
        val_size = 0.1
        test_size = 0.1
        
        X_train, X_validate_test, _, y_validate_test = train_test_split(x, lables, test_size = 0.2, random_state = 42, shuffle=False)
        X_validate, X_test, _, _ = train_test_split(X_validate_test, y_validate_test, test_size = 0.5, random_state = 42, shuffle=False)

    

        # Process your data, create Data objects and save them
        data = Data(
            x=torch.tensor(node,dtype=torch.float32),
            y=torch.tensor(lables, dtype=torch.int64),
            edge_index=torch.tensor(edges_index,dtype=torch.int64),
            edge_attr=torch.tensor(edge_attrs,dtype=torch.float32),
            train_mask=torch.tensor(X_train, dtype=torch.uint8),
            val_mask=torch.tensor(X_validate, dtype=torch.uint8),
            test_mask=torch.tensor(X_test, dtype=torch.uint8)
        )

        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        torch.save((data, self.slices), self.processed_paths[0])

# # Define the root directory where you want to store the processed data
root = "/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/"

# Instantiate your custom dataset class
dataset = CustomCora(root)


data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
for data in data_loader:
    pass



