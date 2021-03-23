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
        pass


class EmbeddingLayer(nn.Module):
    def __init__(self, nunique):
        self.nunique = nunique
        self.emb_dim = int(min(np.ceil(self.nunique/2), 50))
        self.fc1 = nn.Linear()

class PlaygroundModel(nn.Module):
    def __init__(self):
        super(PlaygroundModel, self).__init__()
