"""
Made with love by tcapelle
@wandbcode{pis_course}
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torchvision.transforms as T
from fastprogress import progress_bar
from torch.utils.data import DataLoader
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    Mean,
)

import params
import wandb
from utils import ImageDataset
from utils import get_class_name_in_snake_case as snake_case
from utils import get_data, load_model, set_seed, to_device



# define the default configuration parameters for the experiment
default_cfg = SimpleNamespace(
    image_size=256,
    batch_size=16,
    seed=42,
    model_artifact_name="pyimagesearch/model-registry/Lemon Detector:staging",
    # these are params that are not being changed
    image_column="file_name",
    target_column="mold",
    PROJECT_NAME=params.PROJECT_NAME,
    ENTITY=params.ENTITY,
    PROCESSED_DATA_AT=params.DATA_AT,
)


def main(cfg):
    """Main evaluation loop. This function evaluates the model on the validation and test sets.
    It logs the results to W&B.
    """
    set_seed(cfg.seed)

    run = wandb.init(
        project=cfg.PROJECT_NAME,
        entity=cfg.ENTITY,
        job_type="evaluation",
        tags=["staging"],
    )

    wandb.config.update(cfg)

    # load the data
    df, processed_dataset_dir = get_data(cfg.PROCESSED_DATA_AT, eval=True)

    test_data = df[df["test"] == True]
    val_data = df[df["test"] == False]

    # define the image data transformations, same as the ones used during training
    test_transforms = val_transforms = [
        T.Resize(cfg.image_size),
        T.ToTensor(),
    ]
    val_dataset = ImageDataset(
        val_data,
        processed_dataset_dir,
        image_column=cfg.image_column,
        target_column=cfg.target_column,
        transform=val_transforms,
    )

    test_dataset = ImageDataset(
        test_data,
        processed_dataset_dir,
        image_column=cfg.image_column,
        target_column=cfg.target_column,
        transform=test_transforms,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )
    valid_dataloader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )

    # load the model from the model registry
    model = load_model(cfg.model_artifact_name)

    # move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def cross_entropy(x, y):
        "A flattened version of nn.BCEWithLogitsLoss"
        loss_func = nn.BCEWithLogitsLoss()
        return loss_func(x.squeeze(), y.squeeze().float())

    @torch.inference_mode()
    def evaluate(loader):
        loss_mean = Mean(device=device)
        metrics = [
            BinaryAccuracy(device=device),
            BinaryPrecision(device=device),
            BinaryRecall(device=device),
            BinaryF1Score(device=device),
        ]

        # loop over the data and compute the loss and metrics
        for b in progress_bar(loader, leave=True, total=len(loader)):
            images, labels = to_device(b, device)
            outputs = model(images).squeeze()
            loss = cross_entropy(outputs, labels)
            loss_mean.update(loss)
            for metric in metrics:
                metric.update(outputs, labels.long())

        return loss, metrics

    # evaluate the model on the validation and test sets
    valid_loss, valid_metrics = evaluate(valid_dataloader)
    test_loss, test_metrics = evaluate(test_dataloader)

    # log the validation and test metrics to wandb
    def log_summary(loss, metrics, suffix="valid"):
        wandb.summary[f"{suffix}_loss"] = loss
        for m in metrics:
            wandb.summary[f"{suffix}_{snake_case(m)}"] = m.compute()

    log_summary(valid_loss, valid_metrics, suffix="valid")
    log_summary(test_loss, test_metrics, suffix="test")

    run.finish()


if __name__ == "__main__":
    main(default_cfg)
