from types import SimpleNamespace
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from fastprogress import progress_bar
from torcheval.metrics import (
    Mean,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)

import params
from utils import get_data, set_seed, ImageDataset, load_model, to_device, get_class_name_in_snake_case as snake_case

default_cfg = SimpleNamespace(
    img_size=256,
    bs=16,
    seed=42,
    epochs=2,
    lr=2e-3,
    wd=1e-5,
    arch="resnet18",
    log_model=True,
    log_preds=False,
    # these are params that are not being changed
    image_column="file_name",
    target_column="mold",
    PROJECT_NAME=params.PROJECT_NAME,
    ENTITY=params.ENTITY,
    PROCESSED_DATA_AT=params.DATA_AT,
    model_artifact_name = "wandb_course/model-registry/Lemon Mold Detector:candidate",
)

def main(cfg):

    set_seed(cfg.seed)

    run = wandb.init(
        project=cfg.PROJECT_NAME,
        entity=cfg.ENTITY,
        job_type="evaluation",
        tags=["staging"],
    )

    wandb.config.update(cfg)

    df, processed_dataset_dir = get_data(cfg.PROCESSED_DATA_AT, eval=True)

    test_data = df[df["test"] == True]
    val_data = df[df["test"] == False]

    test_transforms = val_transforms = [
        T.Resize(cfg.img_size),
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
        transform=val_transforms,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.bs, shuffle=False, num_workers=4
    )
    valid_dataloader = DataLoader(
        test_dataset, batch_size=cfg.bs, shuffle=False, num_workers=4
    )
    model = load_model(cfg.model_artifact_name)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def cross_entropy(x, y):
        "A flattened version of nn.BCEWithLogitsLoss"
        loss_func = nn.BCEWithLogitsLoss()
        return loss_func(x.squeeze(), y.squeeze().float())

    @torch.inference_mode()
    def evaluate(loader):
        loss_mean = Mean(device=device)
        metrics = [BinaryAccuracy(device=device),
                BinaryPrecision(device=device),
                BinaryRecall(device=device),
                BinaryF1Score(device=device),
                ]

        for b in progress_bar(loader, leave=True, total=len(loader)):
            images, labels = to_device(b, device)
            outputs = model(images).squeeze()
            loss = cross_entropy(outputs, labels)
            loss_mean.update(loss)
            for metric in metrics:
                metric.update(outputs, labels.long())


        return loss, metrics

    valid_loss, valid_metrics = evaluate(valid_dataloader)
    test_loss, test_metrics   = evaluate(test_dataloader)

    def log_summary(loss, metrics, suffix="valid"):
        wandb.summary[f"{suffix}_loss"] = loss
        for m in metrics:
            wandb.summary[f"{suffix}_{snake_case(m)}"] = m.compute()
    
    log_summary(valid_loss, valid_metrics, suffix="valid")
    log_summary(test_loss, test_metrics, suffix="test")
    
    run.finish()

if __name__ == "__main__":
    main(default_cfg)