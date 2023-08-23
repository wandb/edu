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

def to_snake_case(name):
    "Converts CamelCase to snake_case"
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def get_class_name_in_snake_case(obj):
    "Get the class name of `obj` in snake_case"
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
    "Move `t` to `device`"
    if isinstance(t, (tuple, list)):
        return [_t.to(device) for _t in t]
    elif isinstance(t, torch.Tensor):
        return t.to(device)
    else:
        raise ("Not a Tensor or list of Tensors")


def get_data(PROCESSED_DATA_AT, eval=False):
    """
    Get/Download the datasets from wandb artifacts
    Args:
        PROCESSED_DATA_AT (str): wandb artifact name
        eval (bool, optional): If True, returns test and validation datasets. Defaults to False.
    Returns:
        df (pandas.DataFrame): DataFrame containing image filenames and labels.
        processed_dataset_dir (Path): Directory containing the images.
    """
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
        "Get the image and label at `idx`"
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
    "Get the size of the model in MBs"
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def log_model_preds(test_dl, preds, n=-1, th=0.5):
    """Log the predictions of the model to wandb as a table
    Args:
        test_dl (DataLoader): DataLoader for the test dataset.
        preds (Tensor): Model predictions.
        n (int, optional): Number of samples to log. Defaults to -1 (log all preds).
        th (float, optional): Threshold for the predictions. Defaults to 0.5.
    """
    wandb_table = wandb.Table(columns=["images", "predictions", "groudn_truth"])

    preds = preds[:n]
    for (image, target), pred in progress_bar(
        zip(test_dl.dataset, preds), total=len(preds)
    ):
        wandb_table.add_data(wandb.Image(image), pred > th, target == 1)
    # Log Table
    wandb.log({"predictions": wandb_table})


def save_model(model, model_name, models_folder="models"):
    """Save the model to wandb as an artifact
    Args:
        model (nn.Module): Model to save.
        model_name (str): Name of the model.
        models_folder (str, optional): Folder to save the model. Defaults to "models".
    """
    model_name = f"{wandb.run.id}_{model_name}"
    file_name = Path(f"{models_folder}/{model_name}.pth")
    file_name.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), file_name)
    at = wandb.Artifact(model_name, type="model")
    at.add_file(file_name)
    wandb.log_artifact(at)


def first(iterable, default=None):
    "Returns first element of `iterable` that is not None"
    return next(filter(None, iterable), default)


def load_model(model_artifact_name, eval=True):
    """Load the model from wandb artifacts
    Args:
        model_artifact_name (str): Name of the model artifact.
        eval (bool, optional): If True, sets the model to eval mode. Defaults to True.
    Returns:
        model (nn.Module): Loaded model.
    """
    artifact = wandb.use_artifact(model_artifact_name, type="model")
    model_path = Path(artifact.download()).absolute()

    # recover model info from the registry
    producer_run = artifact.logged_by()
    model_config = SimpleNamespace(
        image_size=producer_run.config["image_size"],
        batch_size=producer_run.config["batch_size"],
        model_arch=producer_run.config["model_arch"],
    )

    model_weights = torch.load(first(model_path.glob("*.pth")))  # get first file
    model = timm.create_model(model_config.model_arch, pretrained=False, num_classes=1)
    model.load_state_dict(model_weights)
    if eval:
        model.eval()
    return model
