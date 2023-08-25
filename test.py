import random
import sys
sys.path.append("/root/Attibute_Social_Network_Embedding")
import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from model import Graphormer
import wandb
from my_graphormer.data.my_dataset import CustomCora
import time
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx
import torch.nn as nn
a = nn.Linear(400, 4)
b =  64
b = torch.tensor((1,0))
e_tensor = a(b)
