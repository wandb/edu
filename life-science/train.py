import torch
import wandb
from tqdm.auto import tqdm

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism

from utils import ConvertToMultiChannelBasedOnBratsClassesd, inference


def main():
    wandb.init(
        project="brain-tumor-segmentation",
        entity="lifesciences",
        job_type="train_baseline",
    )
    config = wandb.config
    config.seed = 0
    config.roi_size = [224, 224, 144]
    config.num_workers = 4
    config.batch_size = 2
    config.model_blocks_down = [1, 2, 2, 4]
    config.model_blocks_up = [1, 1, 1]
    config.model_in_channels = 4
    config.model_out_channels = 3
    config.max_train_epochs = 5
    config.dice_loss_squared_prediction = True
    config.dice_loss_target_onehot = False
    config.dice_loss_apply_sigmoid = True
    config.inference_roi_size = (240, 240, 160)
    config.validation_intervals = 1

    set_determinism(seed=config.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            # Ensure loaded images are in channels-first format
            EnsureChannelFirstd(keys="image"),
            # Ensure the input data to be a PyTorch Tensor or numpy array
            EnsureTyped(keys=["image", "label"]),
            # Convert labels to multi-channels based on brats18 classes
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # Change the input image’s orientation into the specified based on axis codes
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Resample the input images to the specified pixel dimension
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # Augmentation: Crop image with random size or specific size ROI
            RandSpatialCropd(
                keys=["image", "label"], roi_size=config.roi_size, random_size=False
            ),
            # Augmentation: Randomly flip the image on the specified axes
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # Normalize input image intensity
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # Augmentation: Randomly scale the image intensity
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            # Ensure loaded images are in channels-first format
            EnsureChannelFirstd(keys="image"),
            # Ensure the input data to be a PyTorch Tensor or numpy array
            EnsureTyped(keys=["image", "label"]),
            # Convert labels to multi-channels based on brats18 classes
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # Change the input image’s orientation into the specified based on axis codes
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Resample the input images to the specified pixel dimension
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # Normalize input image intensity
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    # Create the dataset for the training split
    # of the brain tumor segmentation dataset
    train_dataset = DecathlonDataset(
        root_dir="./artifacts/decathlon_brain_tumor:v0",
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=False,
        cache_rate=0.0,
        num_workers=config.num_workers,
    )

    # Create the dataset for the validation split
    # of the brain tumor segmentation dataset
    val_dataset = DecathlonDataset(
        root_dir="./artifacts/decathlon_brain_tumor:v0",
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=config.num_workers,
    )

    # create the train_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    # create the val_loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # create model
    model = SegResNet(
        blocks_down=config.model_blocks_down,
        blocks_up=config.model_blocks_up,
        init_filters=config.model_init_filters,
        in_channels=config.model_in_channels,
        out_channels=config.model_out_channels,
        dropout_prob=config.model_dropout_prob,
    ).to(device)

    # create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        config.initial_learning_rate,
        weight_decay=config.weight_decay,
    )

    # create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_train_epochs
    )

    loss_function = DiceLoss(
        smooth_nr=config.dice_loss_smoothen_numerator,
        smooth_dr=config.dice_loss_smoothen_denominator,
        squared_pred=config.dice_loss_squared_prediction,
        to_onehot_y=config.dice_loss_target_onehot,
        sigmoid=config.dice_loss_apply_sigmoid,
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    postprocessing_transforms = Compose(
        [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )

    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    wandb.define_metric("epoch/epoch_step")
    wandb.define_metric("epoch/*", step_metric="epoch/epoch_step")
    wandb.define_metric("batch/batch_step")
    wandb.define_metric("batch/*", step_metric="batch/batch_step")
    wandb.define_metric("validation/validation_step")
    wandb.define_metric("validation/*", step_metric="validation/validation_step")

    batch_step = 0
    validation_step = 0
    metric_values = []
    metric_values_tumor_core = []
    metric_values_whole_tumor = []
    metric_values_enhanced_tumor = []

    epoch_progress_bar = tqdm(range(config.max_train_epochs), desc="Training:")

    for epoch in epoch_progress_bar:
        model.train()
        epoch_loss = 0

        total_batch_steps = len(train_dataset) // train_loader.batch_size
        batch_progress_bar = tqdm(train_loader, total=total_batch_steps, leave=False)

        # Training Step
        for batch_data in batch_progress_bar:
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            batch_progress_bar.set_description(f"train_loss: {loss.item():.4f}:")
            ## Log batch-wise training loss to W&B
            wandb.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
            batch_step += 1

        epoch_loss /= total_batch_steps
        ## Log batch-wise training loss and learning rate to W&B
        wandb.log(
            {
                "epoch/epoch_step": epoch,
                "epoch/mean_train_loss": epoch_loss,
                "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
            }
        )
        lr_scheduler.step()
        epoch_progress_bar.set_description(f"Training: train_loss: {epoch_loss:.4f}:")

        # Validation and model checkpointing step
        if (epoch + 1) % config.validation_intervals == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = inference(model, val_inputs, config.roi_size)
                    val_outputs = [
                        postprocessing_transforms(i)
                        for i in decollate_batch(val_outputs)
                    ]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric_values.append(dice_metric.aggregate().item())
                metric_batch = dice_metric_batch.aggregate()
                metric_values_tumor_core.append(metric_batch[0].item())
                metric_values_whole_tumor.append(metric_batch[1].item())
                metric_values_enhanced_tumor.append(metric_batch[2].item())
                dice_metric.reset()
                dice_metric_batch.reset()

                # Log validation metrics to W&B dashboard.
                wandb.log(
                    {
                        "validation/validation_step": validation_step,
                        "validation/mean_dice": metric_values[-1],
                        "validation/mean_dice_tumor_core": metric_values_tumor_core[-1],
                        "validation/mean_dice_whole_tumor": metric_values_whole_tumor[
                            -1
                        ],
                        "validation/mean_dice_enhanced_tumor": metric_values_enhanced_tumor[
                            -1
                        ],
                    }
                )
                validation_step += 1


if __name__ == "__main__":
    main()
