import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from fastprogress import progress_bar
from PIL import Image

import wandb

VAL_DATA_AT = "fastai/fmnist_pt/validation_data:latest"


def to_snake_case(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def get_class_name_in_snake_case(obj):
    class_name = obj.__class__.__name__
    return to_snake_case(class_name)


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s % (2**32 - 1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_device(t, device):
    if isinstance(t, (tuple, list)):
        return [_t.to(device) for _t in t]
    elif isinstance(t, torch.Tensor):
        return t.to(device)
    else:
        raise ("Not a Tensor or list of Tensors")


def get_data(PROCESSED_DATA_AT, eval=False):
    "Get/Download the datasets"
    processed_data_at = wandb.use_artifact(PROCESSED_DATA_AT)
    processed_dataset_dir = Path(processed_data_at.download())
    df = pd.read_csv(processed_dataset_dir / "data_split.csv")
    if eval:  # for eval we need test and validation datasets only
        df = df[df.stage != "train"].reset_index(drop=True)
        df["test"] = df.stage == "test"
    else:
        df = df[df.stage != "test"].reset_index(drop=True)
        df["valid"] = df.stage == "valid"
    return df, processed_dataset_dir


class ImageDataset:
    def __init__(
        self,
        dataframe,
        root_dir,
        transform=None,
        image_column="file_name",
        target_column="mold",
    ):
        """
        Args:
            dataframe (pandas.DataFrame): DataFrame containing image filenames and labels.
            root_dir (string): Directory containing the images.
            transform (callable, optional): Optional transform to be applied on an image sample.
            image_column (string, optional): Name of the column containing the image filenames.
            target_column (string, optional): Name of the column containing the labels.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        if transform is not None:
            self.transform = T.Compose(transform)
        self.image_column = image_column
        self.target_column = target_column

    def __len__(self):
        return len(self.dataframe)

    def loc(self, idx):
        idx_of_image_column = self.dataframe.columns.get_loc(self.image_column)
        idx_of_target_column = self.dataframe.columns.get_loc(self.target_column)
        x = self.dataframe.iloc[idx, idx_of_image_column]
        y = self.dataframe.iloc[idx, idx_of_target_column]
        return x, y

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, label = self.loc(idx)
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


def model_size(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def log_model_preds(test_dl, preds, n=5, th=0.5):
    wandb_table = wandb.Table(columns=["images", "predictions", "groudn_truth"])

    preds = preds[:n]
    for (image, target), pred in progress_bar(
        zip(test_dl.dataset, preds), total=len(preds)
    ):
        wandb_table.add_data(wandb.Image(image), pred > th, target == 1)
    # Log Table
    wandb.log({"predictions": wandb_table})


def save_model(model, model_name):
    "Save the model to wandb"
    model_name = f"{wandb.run.id}_{model_name}"
    models_folder = Path("models")
    if not models_folder.exists():
        models_folder.mkdir()
    torch.save(model.state_dict(), models_folder / f"{model_name}.pth")
    at = wandb.Artifact(model_name, type="model")
    at.add_file(f"models/{model_name}.pth")
    wandb.log_artifact(at)


def first(iterable, default=None):
    "Returns first element of `iterable` that is not None"
    return next(filter(None, iterable), default)


def load_model(model_artifact_name, eval=True):
    "Load the model from wandb"

    artifact = wandb.use_artifact(model_artifact_name, type="model")
    model_path = Path(artifact.download()).absolute()

    # recover model info from the registry
    producer_run = artifact.logged_by()
    model_config = SimpleNamespace(
        img_size=producer_run.config["img_size"],
        bs=producer_run.config["bs"],
        arch=producer_run.config["arch"],
    )

    model_weights = torch.load(first(model_path.glob("*.pth")))  # get first file
    model = timm.create_model(model_config.arch, pretrained=False, num_classes=1)
    model.load_state_dict(model_weights)
    if eval:
        model.eval()
    return model
