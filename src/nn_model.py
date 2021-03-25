import sys
import os
import pandas as pd
import numpy as np
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score

from torch.utils.data import DataLoader, Dataset


class PlaygroundData(Dataset):
    def __init__(self, data=None, path=None, train=True):
        if data is not None:
            self.data = data
        else:
            self.data = pd.read_csv(path)
        self.catcols = [col for col in self.data.columns if col.endswith("le")]
        self.contcols = [col for col in self.data.columns if col.startswith("cont")]
        self.features = self.catcols + self.contcols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cat_features = self.data[self.catcols].iloc[idx]
        cont_features = self.data[self.contcols].iloc[idx]

        label = self.data.target.iloc[idx]
        return (
            cat_features.values.astype(np.int32),
            cont_features.values.astype(np.float32),
            label,
        )

    @classmethod
    def from_df(cls, df):
        return cls(data=df)

    @staticmethod
    def embed_dim(n):
        return int(min(np.ceil(n / 2), 50))

    def embedding_sizes(self):
        sizes = []

        for col in self.catcols:
            nunique = self.data[col].max()
            emb_dim = self.embed_dim(nunique)
            sizes.append((nunique + 1, emb_dim))

        return sizes


class PlaygroundModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super(PlaygroundModel, self).__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embedding, embedding_dim)
                for num_embedding, embedding_dim in embedding_sizes
            ]
        )
        self.n_emb = sum(emb.embedding_dim for emb in self.embeddings)
        self.n_cont = n_cont

        self.cont1 = nn.Linear(n_cont, 256)
        self.cont2 = nn.Linear(256, 256)
        self.cont3 = nn.Linear(256, 256)

        self.fc1 = nn.Linear(self.n_emb + 256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

        self.emb_drops = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def forward(self, x_cat, x_cont):
        x = [emb(x_cat[:, i]) for i, emb, in enumerate(self.embeddings)]
        x = torch.cat(x, dim=1)
        x = self.emb_drops(x)
        x_cont = self.bn1(x_cont)
        x_cont = self.cont1(x_cont)
        x_cont = F.relu(x_cont)
        x_cont = self.cont2(x_cont)
        x_cont = F.relu(x_cont)
        x_cont = self.bn2(x_cont)
        x_cont = self.cont3(x_cont)
        x_cont = F.relu(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.fc3(x)
        return x

    def predict_proba(self, x_cat, x_cont):
        x = self.forward(x_cat, x_cont)
        return nn.Softmax()(x)


def run(fold, epochs=10):
    df = pd.read_csv(config.TRAIN_DATA)

    train = PlaygroundData.from_df(df.loc[df.kfold != fold])
    valid = PlaygroundData.from_df(df.loc[df.kfold == fold])

    train_dl = DataLoader(train, batch_size=1024, shuffle=True)
    valid_dl = DataLoader(valid, batch_size=2048, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = PlaygroundModel(train.embedding_sizes(), len(train.contcols))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0
        model.train()
        with tqdm(train_dl, unit="batch") as tepoch:
            for idx, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}.")
                x_cat, x_cont, y = batch
                x_cat.to(device)
                x_cont.to(device)
                y.to(device)

                outputs = model(x_cat, x_cont)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss)
                if idx % 10 == 9:
                    tepoch.set_postfix(loss=running_loss / (10 * idx))

        model.eval()
        with torch.no_grad():
            with tqdm(valid_dl, unit="batch") as vepoch:
                running_auc = 0
                for idx, batch in enumerate(vepoch):
                    vepoch.set_description(f"Validation")
                    x_cat, x_cont, y = batch
                    x_cat.to(device)
                    x_cont.to(device)
                    y.to(device)

                    batch_proba = (
                        model.predict_proba(x_cat, x_cont).detach().numpy()[:, 1]
                    )
                    auc_score = roc_auc_score(y.numpy(), batch_proba)
                    running_auc += auc_score
                    if idx > 1:
                        vepoch.set_postfix(AUC=running_auc / (idx))


# class DenoisingAutoEncoder(nn.Module):
#     def __init__(self):
#         super(DenoisingAutoEncoder, self).__init__()
#         self.fc1 = nn.Linear(11, 1500)
#         self.fc2 = nn.Linear(1500, 1500)
#         self.dp1 = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(1500, 1500)
#         self.fc4 = nn.Linear(1500, 11)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.dp1(x)
#         x = self.fc3(x)
#         x = F.relu(x)
#         x = self.fc4(x)
#         return x

if __name__ == "__main__":
    run(0)