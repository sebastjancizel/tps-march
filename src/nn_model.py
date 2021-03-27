import pandas as pd
import numpy as np
import config
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F


from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class PlaygroundData(Dataset):
    def __init__(
        self,
        data=None,
        path=None,
    ):
        if data is not None:
            self.data = data
        else:
            self.data = pd.read_csv(path)
        self.catcol_names = [col for col in self.data.columns if col.endswith("le")]
        self.contcol_names = [
            col for col in self.data.columns if col.startswith("cont")
        ]
        self.features = self.catcol_names + self.contcol_names
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.catcols = torch.tensor(
            self.data[self.catcol_names].values, device=self.device, dtype=torch.long
        )
        self.contcols = torch.tensor(
            self.data[self.contcol_names].values,
            device=self.device,
            dtype=torch.float32,
        )
        self.target = torch.tensor(
            self.data.target.values, device=self.device, dtype=torch.long
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_cat = self.catcols[idx, :]
        x_cont = self.contcols[idx, :]
        y = self.target[idx]
        return x_cat, x_cont, y

    @classmethod
    def from_df(cls, df):
        return cls(data=df)

    @staticmethod
    def embed_dim(n):
        """ Calculates the embedding dimension given the number of categories """
        return int(min(np.ceil(n / 2), 50))

    def embedding_sizes(self):
        sizes = []

        for col in self.catcol_names:
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

        self.cont1 = nn.Linear(n_cont, 64)
        self.cont2 = nn.Linear(64, 64)
        self.cont3 = nn.Linear(64, 64)

        self.fc1 = nn.Linear(self.n_emb + 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.emb_drops = nn.Dropout(0.3)
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
        return nn.Softmax(-1)(x)


def fold_split(df, fold):
    train = PlaygroundData.from_df(df.loc[df.kfold != fold])
    valid = PlaygroundData.from_df(df.loc[df.kfold == fold])
    return train, valid


def train_loop(train_dl, model, optimizer, criterion, epoch, writer=None):

    model.train()

    training_loss = utils.AverageMeter(name="loss")
    with tqdm(train_dl, unit="batch") as tepoch:
        for batch in tepoch:
            optimizer.zero_grad()

            tepoch.set_description(f"Epoch {epoch}.")
            x_cat, x_cont, y = batch

            outputs = model(x_cat, x_cont)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            training_loss.update(loss.item(), n=x_cat.shape[0])

            tepoch.set_postfix(Loss=training_loss.avg)

            if writer is not None:
                writer.add_scalar("Loss/train", training_loss.avg)


def eval_loop(valid_dl, model, writer=None):

    model.eval()

    valid_auc = utils.AverageMeter(name="AUC")
    with torch.no_grad():
        with tqdm(valid_dl, unit="batch") as vepoch:
            for batch in vepoch:
                vepoch.set_description(f"Validation")
                x_cat, x_cont, y = batch

                batch_proba = (
                    model.predict_proba(x_cat, x_cont).detach().cpu().numpy()[:, 1]
                )
                auc_score = roc_auc_score(y.cpu().numpy(), batch_proba)
                valid_auc.update(auc_score, n=x_cat.shape[0])
                vepoch.set_postfix(AUC=valid_auc.avg)
                if writer is not None:
                    writer.add_scalar("AUC", valid_auc.avg)

    return valid_auc


def run(fold, epochs=10, bs=512, lr=1e-3, lr_decay=0.95):
    df = pd.read_csv(config.TRAIN_DATA)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train, valid = fold_split(df, fold)

    train_dl = DataLoader(train, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid, batch_size=4096, shuffle=False)

    model = PlaygroundModel(train.embedding_sizes(), 11)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: lr_decay * epoch
    )

    # Logging setup
    time = datetime.now().strftime("%Y-%m-%d_%H:%M")
    params = f"bs={bs}_lr={lr}_lr-decay={lr_decay}__{time}"
    writer = SummaryWriter(log_dir=config.LOG_DIR / params / f"Fold={fold}")

    for epoch in range(epochs):
        train_loop(train_dl, model, optimizer, criterion, epoch, writer=writer)
        auc = eval_loop(valid_dl, model, writer=writer)
        scheduler.step()

    model_export_path = config.MODEL_DIR / params
    model_export_path.mkdir(parents=True, exist_ok=True)

    torch.save(model, model_export_path / f"Fold={fold}_AUC={auc.avg}.pth")


if __name__ == "__main__":
    run(0)