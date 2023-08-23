"""
Made with love by tcapelle
@wandbcode{pis_course}
"""

import argparse
from types import SimpleNamespace
from fastprogress import progress_bar
import timm
import wandb
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    Mean,
)
from utils import (
    get_data,
    ImageDataset,
    set_seed,
    to_device,
    save_model,
    log_model_preds,
    get_class_name_in_snake_case as snake_case,
)
import params

# Set the default configuration parameters for the experiment
default_cfg = SimpleNamespace(
    image_size=256,                        # Image size
    batch_size=16,                         # Batch size
    seed=42,                               # Random seed
    epochs=10,                             # Number of training epochs
    learning_rate=2e-3,                    # Learning rate
    weight_decay=1e-5,                     # Weight decay
    model_arch="resnet18",                 # Timm backbone architecture
    log_model=False,                       # Whether or not to log the model to Wandb
    log_preds=False,                       # Whether or not to log the model predictions to Wandb
    # these are params that are not being changed
    image_column="file_name",              # The name of the column containing the image file names
    target_column="mold",                  # The name of the column containing the target variable
    PROJECT_NAME=params.PROJECT_NAME,      # The name of the Wandb project
    ENTITY=params.ENTITY,                  # The Wandb username or organization name
    PROCESSED_DATA_AT=params.DATA_AT,      # The path to the directory containing the preprocessed data
)

# Define the image data transformations
transforms = {
    "train": [T.Resize(default_cfg.image_size), T.ToTensor(), T.RandomHorizontalFlip()],
    "valid": [T.Resize(default_cfg.image_size), T.ToTensor()],
}

# Override the default configuration parameters with any command-line arguments
def parse_args(default_cfg):
    "Overriding default argments"
    parser = argparse.ArgumentParser(description="Process hyper-parameters")
    parser.add_argument("--image_size", type=int, default=default_cfg.image_size, help="image size")
    parser.add_argument("--batch_size", type=int, default=default_cfg.batch_size, help="batch size")
    parser.add_argument("--seed", type=int, default=default_cfg.seed, help="random seed")
    parser.add_argument("--epochs",type=int,default=default_cfg.epochs, help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=default_cfg.learning_rate, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=default_cfg.weight_decay, help="weight decay")
    parser.add_argument("--model_arch", type=str, default=default_cfg.model_arch, help="timm backbone architecture")
    parser.add_argument("--log_model", action="store_true", help="log model to wandb")
    parser.add_argument(
        "--log_preds", action="store_true", help="log model predictions to wandb"
    )
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(default_cfg, k, v)

# Define the ClassificationTrainer class for training the model
class ClassificationTrainer:
    """
    A class for training a classification model. It is used to train a model 
    on a training set and validate it on a validation set. This class is
    inspired by the Keras API.

    Args:
        train_dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader for the training set
        valid_dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader for the validation set
        model (torch.nn.Module): A PyTorch model
        metrics (list): A list of metrics to be used for training and validation, 
            we are using torcheval.metrics
        device (str): The device to be used for training, either "cpu" or "cuda"
    """
    def __init__(
        self, train_dataloader, valid_dataloader, model, metrics, device="cuda"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.train_metrics = [m(device=self.device) for m in metrics]
        self.valid_metrics = [m(device=self.device) for m in metrics]
        self.loss = Mean(device=device)

    def loss_func(self, x, y):
        "A flattened version of nn.BCEWithLogitsLoss"
        loss_func = nn.BCEWithLogitsLoss()
        return loss_func(x.squeeze(), y.squeeze().float())

    def compile(self, epochs=5, learning_rate=2e-3, weight_decay=0.01):
        "Keras style compile method"
        self.epochs = epochs
        self.optim = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.schedule = OneCycleLR(
            self.optim,
            max_lr=learning_rate,
            pct_start=0.1,
            total_steps=epochs * len(self.train_dataloader),
        )

    def reset_metrics(self):
        "Reset the metrics after each epoch"
        self.loss.reset()
        for m in self.train_metrics:
            m.reset()
        for m in self.valid_metrics:
            m.reset()

    def train_step(self, loss):
        "Perform a single training step"
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.schedule.step()
        return loss

    def one_epoch(self, train=True):
        "Perform a single epoch of training or validation"
        if train:
            self.model.train()
            dl = self.train_dataloader
        else:
            self.model.eval()
            dl = self.valid_dataloader
        pbar = progress_bar(dl, leave=False)
        preds = []
        for b in pbar:
            with torch.inference_mode() if not train else torch.enable_grad():
                images, labels = to_device(b, self.device)
                preds_b = self.model(images).squeeze()
                loss = self.loss_func(preds_b, labels)
                self.loss.update(loss)
                preds.append(preds_b)
                if train:
                    self.train_step(loss)
                    for m in self.train_metrics:
                        m.update(preds_b, labels.long())
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "learning_rate": self.schedule.get_last_lr()[0],
                        }
                    )
                else:
                    for m in self.valid_metrics:
                        m.update(preds_b, labels.long())
            pbar.comment = f"train_loss={loss.item():2.3f}"

        return torch.cat(preds, dim=0), self.loss.compute()

    def print_metrics(self, epoch, train_loss, val_loss):
        "Print the metrics after each epoch"
        print(f"Epoch {epoch+1}/{self.epochs} - train_loss: {train_loss.item():2.3f} - val_loss: {val_loss.item():2.3f}")

    # Fit the model
    def fit(self, log_preds=False):
        "Fit the model for the specified number of epochs"
        for epoch in progress_bar(range(self.epochs), total=self.epochs, leave=True):
            # train epoch
            _, train_loss = self.one_epoch(train=True)
            wandb.log({f"train_{snake_case(m)}": m.compute() for m in self.train_metrics})

            ## validation epoch
            val_preds, val_loss = self.one_epoch(train=False)
            wandb.log(
                {f"valid_{snake_case(m)}": m.compute() for m in self.valid_metrics},
                commit=False,
            )
            wandb.log({"valid_loss": val_loss.item()}, commit=False)
            self.print_metrics(epoch, train_loss, val_loss)
            self.reset_metrics()
        if log_preds:
            log_model_preds(self.valid_dataloader, val_preds)

# Train the model with the specified configurations
def train(cfg):
    "Train the model"
    with wandb.init(
        project=cfg.PROJECT_NAME, entity=cfg.ENTITY, job_type="training", config=cfg
    ):
        set_seed(cfg.seed)

        cfg = wandb.config
        df, processed_dataset_dir = get_data(cfg.PROCESSED_DATA_AT)

        # Create training and validation datasets
        train_ds = ImageDataset(
            df[~df.valid],
            processed_dataset_dir,
            image_column=cfg.image_column,
            target_column=cfg.target_column,
            transform=transforms["train"],
        )

        valid_ds = ImageDataset(
            df[df.valid],
            processed_dataset_dir,
            image_column=cfg.image_column,
            target_column=cfg.target_column,
            transform=transforms["valid"],
        )

        # Define training and validation dataloaders
        train_dataloader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=6
        )
        valid_dataloader = DataLoader(
            valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4
        )

        # Create the model using timm library. We will use a pretrained model.
        model = timm.create_model(cfg.model_arch, pretrained=True, num_classes=1)

        # Define the trainer object
        trainer = ClassificationTrainer(
            train_dataloader,
            valid_dataloader,
            model,
            metrics=[BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score],
            device="cuda",
        )
        # Setup the optimizer and loss function
        trainer.compile(epochs=cfg.epochs, learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # Fit the model
        trainer.fit(log_preds=cfg.log_preds)
        if cfg.log_model:
            save_model(trainer.model, cfg.model_arch)

if __name__ == "__main__":
    parse_args(default_cfg)
    train(default_cfg)
