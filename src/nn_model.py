import sys
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


class PlaygroundData(DataLoader):
    def __init__(self, path):
        super(PlaygroundData, self).__init__()


class EmbeddingLayer(nn.Module):
    def __init__(self, nunique):
        self.nunique = nunique
        self.emb_dim = int(min(np.ceil(self.nunique / 2), 50))
        self.fc1 = nn.Linear()


class PlaygroundModel(nn.Module):
    def __init__(self):
        super(PlaygroundModel, self).__init__()


class DenoisingAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoEncoder, self).__init__()
        self.fc1 = nn.Linear(11, 1500)
        self.fc2 = nn.Linear(1500, 1500)
        self.dp1 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1500, 1500)
        self.fc4 = nn.Linear(1500, 11)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
