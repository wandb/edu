import argparse
from types import SimpleNamespace
from fastprogress import progress_bar

import timm

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torchvision.transforms as T

from torcheval.metrics import (
    Mean,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
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


default_cfg = SimpleNamespace(
    img_size=256,
    bs=16,
    seed=42,
    epochs=10,
    lr=2e-3,
    wd=1e-5,
    arch="convnext_tiny",
    log_model=True,
    log_preds=False,
    # these are params that are not being changed
    image_column="file_name",
    target_column="mold",
    PROJECT_NAME=params.PROJECT_NAME,
    ENTITY=params.ENTITY,
    PROCESSED_DATA_AT=params.DATA_AT,
)

tfms = {
    "train": [T.Resize(default_cfg.img_size), T.ToTensor(), T.RandomHorizontalFlip()],
    "valid": [T.Resize(default_cfg.img_size), T.ToTensor()],
}


# optional
def parse_args(default_cfg):
    "Overriding default argments"
    parser = argparse.ArgumentParser(description="Process hyper-parameters")
    parser.add_argument("--img_size", type=int, default=default_cfg.img_size, help="image size")
    parser.add_argument("--bs", type=int, default=default_cfg.bs, help="batch size")
    parser.add_argument("--seed", type=int, default=default_cfg.seed, help="random seed")
    parser.add_argument("--epochs",type=int,default=default_cfg.epochs, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=default_cfg.lr, help="learning rate")
    parser.add_argument("--wd", type=float, default=default_cfg.wd, help="weight decay")
    parser.add_argument("--arch", type=str, default=default_cfg.arch, help="timm backbone architecture")
    parser.add_argument("--log_model", action="store_true", help="log model to wandb")
    parser.add_argument("--log_preds", action="store_true", help="log model predictions to wandb")
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(default_cfg, k, v)


class ClassificationTrainer:
    def __init__(
        self, train_dataloader, valid_dataloader, model, metrics, device="cuda"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.train_metrics = [m(device=self.device) for m in metrics]
        self.valid_metrics = [m(device=self.device) for m in metrics]
        self.loss = Mean()

    def loss_func(self, x, y):
        "A flattened version of nn.BCEWithLogitsLoss"
        loss_func = nn.BCEWithLogitsLoss()
        return loss_func(x.squeeze(), y.squeeze().float())

    def compile(self, epochs=5, lr=2e-3, wd=0.01):
        "Keras style compile method"
        self.epochs = epochs
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.schedule = OneCycleLR(
            self.optim,
            max_lr=lr,
            pct_start=0.1,
            total_steps=epochs * len(self.train_dataloader),
        )

    def reset_metrics(self):
        self.loss.reset()
        for m in self.train_metrics:
            m.reset()
        for m in self.valid_metrics:
            m.reset()

    def train_step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.schedule.step()
        return loss

    def one_epoch(self, train=True):
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
                self.loss.update(loss.detach().cpu(), weight=len(images))
                preds.append(preds_b)
                if train:
                    self.train_step(loss)
                    for m in self.train_metrics:
                        m.update(preds_b, labels.long())
                    wandb.log({"train_loss": loss.item(),
                               "learning_rate": self.schedule.get_last_lr()[0]})
                else:
                    for m in self.valid_metrics:
                        m.update(preds_b, labels.long())
            pbar.comment = f"train_loss={loss.item():2.3f}"

        return torch.cat(preds, dim=0), self.loss.compute()

    def print_metrics(self, epoch, train_loss, val_loss):
        print(f"Epoch {epoch+1}/{self.epochs} - train_loss: {train_loss.item():2.3f} - val_loss: {val_loss.item():2.3f}")
    
    def fit(self, log_preds=False):      
        for epoch in progress_bar(range(self.epochs), total=self.epochs, leave=True):
            _, train_loss = self.one_epoch(train=True)
            wandb.log({f"train_{snake_case(m)}": m.compute() for m in self.train_metrics})
                            
            ## validation
            val_preds, val_loss = self.one_epoch(train=False)
            wandb.log({f"valid_{snake_case(m)}": m.compute() for m in self.valid_metrics}, commit=False)
            wandb.log({"valid_loss": val_loss.item()}, commit=False)
            self.print_metrics(epoch, train_loss, val_loss)
            self.reset_metrics()
            if log_preds:
                log_model_preds(self.valid_dataloader, val_preds)


def train(cfg):
    with wandb.init(
        project=cfg.PROJECT_NAME, entity=cfg.ENTITY, job_type="training", config=cfg
    ):
        set_seed(cfg.seed)

        cfg = wandb.config
        df, processed_dataset_dir = get_data(cfg.PROCESSED_DATA_AT)

        train_ds = ImageDataset(
            df[~df.valid],
            processed_dataset_dir,
            image_column=cfg.image_column,
            target_column=cfg.target_column,
            transform=tfms["train"],
        )

        valid_ds = ImageDataset(
            df[df.valid],
            processed_dataset_dir,
            image_column=cfg.image_column,
            target_column=cfg.target_column,
            transform=tfms["valid"],
        )

        train_dataloader = DataLoader(
            train_ds, batch_size=cfg.bs, shuffle=True, num_workers=4
        )
        valid_dataloader = DataLoader(
            valid_ds, batch_size=cfg.bs, shuffle=False, num_workers=4
        )

        model = timm.create_model(cfg.arch, pretrained=False, num_classes=1)

        trainer = ClassificationTrainer(
            train_dataloader,
            valid_dataloader,
            model,
            metrics=[BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score],
            device="cuda",
        )
        trainer.compile(epochs=cfg.epochs, lr=cfg.lr, wd=cfg.wd)

        trainer.fit(log_preds=cfg.log_preds)
        if cfg.log_model:
            save_model(trainer.model, cfg.arch)


if __name__ == "__main__":
    parse_args(default_cfg)
    train(default_cfg)
