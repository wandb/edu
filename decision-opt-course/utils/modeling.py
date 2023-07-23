import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import wandb

wandb_project = "decision_opt_bimbo"

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


def make_model(data, sample_size=500_000, run_name="bimbo_model"):
    num_unique_vals = {col: data[col].nunique() for col in categorical_cols}
    data = data.sample(sample_size, random_state=0)

    # Split into features and target
    X_categorical = data[categorical_cols].values
    X_numerical = data[numerical_cols].values
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_categorical = encoder.fit_transform(X_categorical)
    y = data["Demanda_uni_equil"].values

    # Split into training and validation sets
    X_train_cat, X_val_cat, y_train, y_val = train_test_split(
        X_categorical, y, test_size=0.2, random_state=0
    )
    X_train_num, X_val_num, _, _ = train_test_split(
        X_numerical, y, test_size=0.2, random_state=0
    )

    class BimboDataset(Dataset):
        def __init__(self, X_cat, X_num, y):
            self.X_cat = [
                torch.tensor(X_cat[:, i], dtype=torch.long)
                for i in range(X_cat.shape[1])
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

    # Create Datasets and DataLoaders
    train_dataset = BimboDataset(X_train_cat, X_train_num, y_train)
    val_dataset = BimboDataset(X_val_cat, X_val_num, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    def train_model(loss_fn, num_epochs=2):
        model = SimpleModel(num_unique_vals)
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        # Training loop
        for epoch in range(num_epochs):
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
            # Validation loop
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
            r2 = r2_score(val_targets, val_preds)
        return model

    loss = nn.MSELoss()
    mse_model = train_model(loss, num_epochs=5)
    return mse_model, encoder
