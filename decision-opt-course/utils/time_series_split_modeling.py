import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
import wandb

wandb_project = "bimbo_drift_check"

categorical_cols = [
    "Agencia_ID",
    "Canal_ID",
    "Ruta_SAK",
    "Cliente_ID",
    "Producto_ID",
]

numerical_cols = [
    "Venta_uni_hoy",
    "Venta_hoy",
]


class SimpleModel(nn.Module):
    def __init__(self, num_unique_vals, hidden_size=128, num_features=2):
        super(SimpleModel, self).__init__()
        embedding_size = 50
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_unique_vals[col], embedding_size)
                for col in categorical_cols
            ]
        )
        self.num_layer = nn.Linear(
            num_features, embedding_size
        )  # define a linear layer for numerical inputs
        self.fc1 = nn.Linear(
            embedding_size * len(num_unique_vals) + embedding_size, hidden_size
        )  # add embedding_size to input features of fc1 to account for the numerical inputs
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x_cat, x_num):
        x_cat = [
            embedding(x_i.clip(0)) for x_i, embedding in zip(x_cat, self.embeddings)
        ]
        x_cat = torch.cat(x_cat, dim=-1)
        x_num = torch.relu(self.num_layer(x_num))
        x = torch.cat([x_cat, x_num], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        return x


def train(train_dataset, val_dataset, num_unique_vals):
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    num_epochs = 5
    loss_fn = nn.MSELoss()
    model = SimpleModel(num_unique_vals)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Training loop
    for _ in range(num_epochs):
        model.train()
        train_loss = 0.0
        for (inputs_cat, inputs_num), targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs_cat, inputs_num).squeeze()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

    # Calculate validation score
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for (inputs_cat, inputs_num), targets in val_loader:
            outputs = model(inputs_cat, inputs_num).squeeze()
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
            val_preds.extend(outputs.tolist())
            val_targets.extend(targets.tolist())
    val_loss /= len(val_loader)
    return model, val_loss


class BimboDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = [
            torch.tensor(X_cat[:, i], dtype=torch.long) for i in range(X_cat.shape[1])
        ]
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_cat = [x[idx] for x in self.X_cat]
        # x_cat = self.X_cat[idx]
        x_num = self.X_num[idx]
        y = self.y[idx]
        return (x_cat, x_num), y  # return inputs as a tuple


def make_models(data):
    num_unique_vals = {col: data[col].nunique() for col in categorical_cols}
    data = data.sort_values("Semana")

    # Split into features and target
    X_categorical = data[categorical_cols].values
    X_numerical = data[numerical_cols].values
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_categorical = encoder.fit_transform(X_categorical)
    y = data["Demanda_uni_equil"].values

    weeks = sorted(data["Semana"].unique())

    run = wandb.init(project=wandb_project)
    model_list = []

    for training_week in weeks[:-1]:
        val_week = training_week + 1
        train_index = data[data["Semana"] == training_week].index
        val_index = data[data["Semana"] == val_week].index

        X_train_cat, X_val_cat = X_categorical[train_index], X_categorical[val_index]
        X_train_num, X_val_num = X_numerical[train_index], X_numerical[val_index]
        y_train, y_val = y[train_index], y[val_index]
        train_dataset = BimboDataset(X_train_cat, X_train_num, y_train)
        val_dataset = BimboDataset(X_val_cat, X_val_num, y_val)

        model, val_loss = train(train_dataset, val_dataset, num_unique_vals)
        metrics = {
            "training_data_week": training_week,
            "validation_data_week": val_week,
            "validation_loss": val_loss,
        }

        wandb.log(metrics)
        print(metrics)
        model_list.append(model)
    run.finish()
    return model_list, encoder
